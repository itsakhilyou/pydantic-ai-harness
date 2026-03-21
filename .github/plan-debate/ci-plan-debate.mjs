#!/usr/bin/env node
/**
 * CI-mode plan-debate runner.
 *
 * Runs the same pipeline as the /plan-debate command but non-interactively:
 *   - Models are specified via CLI args or env vars (no interactive picker)
 *   - No editor review step (plan is written directly)
 *   - PR creation is automatic
 *   - Progress is logged to stdout
 *
 * Usage:
 *   node ci-plan-debate.mjs --issue 4723 \
 *     --repo pydantic/pydantic-ai \
 *     --model-a anthropic/claude-opus-4-6 \
 *     --model-b openai/gpt-4.1 \
 *     --cwd /path/to/repo
 *
 * Environment variables:
 *   ANTHROPIC_API_KEY, OPENAI_API_KEY — or use PI's auth storage
 *   GITHUB_TOKEN — for gh CLI (set automatically in GH Actions)
 */

import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB_SEARCH_SCRIPT = path.join(__dirname, "web-search.mjs");

// ─── Argument Parsing ────────────────────────────────────────────────────────

function getArg(name) {
	const idx = process.argv.indexOf(name);
	return idx !== -1 && process.argv[idx + 1] ? process.argv[idx + 1] : null;
}

const ISSUE_NUMBER = parseInt(getArg("--issue") || process.env.ISSUE_NUMBER || "", 10);
const REPO = getArg("--repo") || process.env.REPO || "pydantic/pydantic-ai";
const MODEL_A = getArg("--model-a") || process.env.MODEL_A || "anthropic/claude-opus-4-6";
const MODEL_B = getArg("--model-b") || process.env.MODEL_B || "openai/gpt-4.1";
const CWD = getArg("--cwd") || process.env.CWD || process.cwd();
const SKIP_PR = process.argv.includes("--skip-pr");
const MAX_DEBATE_ROUNDS = parseInt(getArg("--rounds") || "2", 10);
const MAX_LINKED_DEPTH = 3;
const MAX_LINKED_ITEMS = 30;
const MAX_CONCURRENCY = 4;

if (!ISSUE_NUMBER || isNaN(ISSUE_NUMBER)) {
	console.error("Usage: node ci-plan-debate.mjs --issue <number> [--repo owner/repo] [--model-a provider/model] [--model-b provider/model] [--cwd path] [--skip-pr] [--rounds N]");
	process.exit(1);
}

// ─── Logging ─────────────────────────────────────────────────────────────────

function log(msg) {
	const ts = new Date().toISOString().slice(11, 19);
	console.log(`[${ts}] ${msg}`);
}

// ─── Shell Helpers ───────────────────────────────────────────────────────────

function exec(command, cwd = CWD) {
	return new Promise((resolve) => {
		const proc = spawn("bash", ["-c", command], { cwd, stdio: ["ignore", "pipe", "pipe"] });
		let stdout = "";
		let stderr = "";
		proc.stdout.on("data", (d) => (stdout += d.toString()));
		proc.stderr.on("data", (d) => (stderr += d.toString()));
		proc.on("close", (code) => resolve({ stdout, stderr, code: code ?? 1 }));
		proc.on("error", () => resolve({ stdout, stderr, code: 1 }));
	});
}

// ─── Issue Graph Crawler ─────────────────────────────────────────────────────

function extractIssueRefs(text) {
	const refs = new Set();
	for (const m of text.matchAll(/#(\d+)\b/g)) {
		const n = parseInt(m[1], 10);
		if (n > 0 && n < 100000) refs.add(n);
	}
	for (const m of text.matchAll(/GH-(\d+)\b/gi)) {
		const n = parseInt(m[1], 10);
		if (n > 0 && n < 100000) refs.add(n);
	}
	return Array.from(refs);
}

async function fetchItem(number) {
	const issueResult = await exec(
		`gh issue view ${number} --repo ${REPO} --json number,title,body,url,state,labels,comments 2>/dev/null`,
	);
	if (issueResult.code === 0) {
		try {
			const d = JSON.parse(issueResult.stdout);
			return {
				type: "issue", number: d.number, title: d.title || "", body: d.body || "",
				url: d.url || `https://github.com/${REPO}/issues/${number}`,
				state: d.state || "unknown",
				labels: (d.labels || []).map((l) => typeof l === "string" ? l : l.name),
				comments: (d.comments || []).map((c) => `**@${c.author?.login || "unknown"}**:\n${c.body || ""}`),
			};
		} catch { /* fall through */ }
	}

	const prResult = await exec(
		`gh pr view ${number} --repo ${REPO} --json number,title,body,url,state,labels,comments,files 2>/dev/null`,
	);
	if (prResult.code === 0) {
		try {
			const d = JSON.parse(prResult.stdout);
			return {
				type: "pr", number: d.number, title: d.title || "", body: d.body || "",
				url: d.url || `https://github.com/${REPO}/pull/${number}`,
				state: d.state || "unknown",
				labels: (d.labels || []).map((l) => typeof l === "string" ? l : l.name),
				comments: (d.comments || []).map((c) => `**@${c.author?.login || "unknown"}**:\n${c.body || ""}`),
				files: (d.files || []).map((f) => f.path || f),
			};
		} catch { /* ignore */ }
	}
	return null;
}

async function findCrossReferences(issueNumber) {
	const refs = new Set();
	for (const type of ["issues", "prs"]) {
		const r = await exec(`gh search ${type} "${issueNumber}" --repo ${REPO} --json number --limit 20 2>/dev/null`);
		if (r.code === 0) {
			try {
				for (const item of JSON.parse(r.stdout)) {
					if (item.number && item.number !== issueNumber) refs.add(item.number);
				}
			} catch { /* ignore */ }
		}
	}
	const timeline = await exec(
		`gh api repos/${REPO}/issues/${issueNumber}/timeline --paginate --jq '.[].source.issue.number // empty' 2>/dev/null`,
	);
	if (timeline.code === 0) {
		for (const line of timeline.stdout.split("\n")) {
			const num = parseInt(line.trim(), 10);
			if (num > 0 && num !== issueNumber) refs.add(num);
		}
	}
	return Array.from(refs);
}

async function crawlIssueGraph(rootNumber) {
	const visited = new Set();
	const allItems = [];
	const queue = [{ number: rootNumber, depth: 0 }];

	while (queue.length > 0 && allItems.length < MAX_LINKED_ITEMS) {
		const batch = queue.splice(0, MAX_CONCURRENCY);
		const results = await Promise.all(
			batch.map(async ({ number, depth }) => {
				if (visited.has(number)) return null;
				visited.add(number);
				const item = await fetchItem(number);
				if (!item) return null;
				log(`  Fetched ${item.type} #${item.number}: ${item.title} (depth ${depth})`);

				let newRefs = [];
				if (depth < MAX_LINKED_DEPTH) {
					const allText = [item.body, ...item.comments].join("\n");
					const inlineRefs = extractIssueRefs(allText);
					const crossRefs = depth === 0 ? await findCrossReferences(item.number) : [];
					newRefs = [...new Set([...inlineRefs, ...crossRefs])].filter((n) => !visited.has(n) && n !== rootNumber);
				}
				return { item, newRefs, depth };
			}),
		);

		for (const result of results) {
			if (!result) continue;
			allItems.push(result.item);
			for (const ref of result.newRefs) {
				if (!visited.has(ref) && allItems.length + queue.length < MAX_LINKED_ITEMS) {
					queue.push({ number: ref, depth: result.depth + 1 });
				}
			}
		}
	}

	const root = allItems.find((i) => i.number === rootNumber);
	if (!root) throw new Error(`Could not fetch issue #${rootNumber}`);
	const linked = allItems.filter((i) => i.number !== rootNumber);
	return { root, linked, fullContext: formatIssueGraph(root, linked) };
}

function formatIssueGraph(root, linked) {
	let ctx = formatItem(root, "ROOT");
	if (linked.length > 0) {
		ctx += `\n\n${"=".repeat(80)}\nLINKED ISSUES AND PRs (${linked.length})\n${"=".repeat(80)}\n`;
		for (const item of linked) ctx += formatItem(item, "LINKED") + "\n";
	}
	return ctx;
}

function formatItem(item, tag) {
	let ctx = `\n${"─".repeat(80)}\n[${tag}] ${item.type.toUpperCase()} #${item.number}: ${item.title}\nURL: ${item.url}\nState: ${item.state}\n`;
	if (item.labels?.length > 0) ctx += `Labels: ${item.labels.join(", ")}\n`;
	if (item.files?.length > 0) ctx += `Files changed: ${item.files.join(", ")}\n`;
	ctx += `${"─".repeat(80)}\n\n${item.body}\n`;
	if (item.comments?.length > 0) {
		ctx += `\n### Comments (${item.comments.length})\n\n`;
		for (const c of item.comments) ctx += `---\n${c}\n\n`;
	}
	return ctx;
}

// ─── Subagent Runner ─────────────────────────────────────────────────────────

function runSubagent(task, model) {
	return new Promise((resolve) => {
		const args = ["-p", "--no-session", "--model", model, "--tools", "read,bash,grep,find,ls", task];
		const proc = spawn("pi", args, { cwd: CWD, shell: false, stdio: ["ignore", "pipe", "pipe"] });
		let stdout = "";
		let stderr = "";
		proc.stdout.on("data", (d) => (stdout += d.toString()));
		proc.stderr.on("data", (d) => (stderr += d.toString()));
		proc.on("close", (code) => resolve({ output: stdout, exitCode: code ?? 1, stderr }));
		proc.on("error", () => resolve({ output: stdout, exitCode: 1, stderr }));
	});
}

async function runConcurrent(tasks) {
	const results = new Array(tasks.length);
	let completed = 0;
	let nextIndex = 0;

	const workers = Array.from({ length: Math.min(MAX_CONCURRENCY, tasks.length) }, async () => {
		while (true) {
			const i = nextIndex++;
			if (i >= tasks.length) return;
			const { label, task, model } = tasks[i];
			log(`  [${completed}/${tasks.length}] Starting: ${label}`);
			results[i] = await runSubagent(task, model);
			completed++;
			log(`  [${completed}/${tasks.length}] ${results[i].exitCode === 0 ? "✓" : "✗"} ${label} (${(results[i].output.length / 1024).toFixed(1)}KB)`);
		}
	});
	await Promise.all(workers);
	return results;
}

// ─── LLM Complete (via pi -p) ────────────────────────────────────────────────
// Instead of importing pi-ai complete(), use pi itself as the LLM backend.
// This way auth is handled by pi's existing credential chain.

function llmComplete(model, systemPrompt, userMessage) {
	return new Promise((resolve, reject) => {
		const fullPrompt = `${systemPrompt}\n\n---\n\n${userMessage}`;
		const args = ["-p", "--no-session", "--no-tools", "--model", model, fullPrompt];
		const proc = spawn("pi", args, { cwd: CWD, shell: false, stdio: ["ignore", "pipe", "pipe"] });
		let stdout = "";
		let stderr = "";
		proc.stdout.on("data", (d) => (stdout += d.toString()));
		proc.stderr.on("data", (d) => (stderr += d.toString()));
		proc.on("close", (code) => {
			if (code !== 0) reject(new Error(`pi exited ${code}: ${stderr.slice(0, 500)}`));
			else resolve(stdout);
		});
		proc.on("error", (err) => reject(err));
	});
}

// ─── Prompts (same as extension, but inlined) ────────────────────────────────

const CODEBASE_RESEARCH_PROMPT = `You are a senior software engineer doing a deep-dive investigation of a codebase for an implementation plan.

You have a web search helper for fetching external resources:

  node ${WEB_SEARCH_SCRIPT} fetch "https://some-docs-url.com/page"
  node ${WEB_SEARCH_SCRIPT} gh-readme owner/repo
  node ${WEB_SEARCH_SCRIPT} gh-file owner/repo path/to/file.py

You have been given an issue with its full discussion context (including all linked issues, PRs, and comments).
Your job: exhaustively investigate the codebase to gather every fact that could be relevant.

CRITICAL RULES:
- For EVERY claim you make, include a citation: file path + line number, or a URL.
- Do NOT guess or assume. If you're not sure, say "NOT VERIFIED".
- Read actual code. Trace actual call chains. Check actual tests.
- Do NOT propose solutions. Only gather facts.

Investigation strategy:
1. Start from the entry points mentioned in the issue
2. grep/find to locate all relevant code
3. Read key files — the actual implementation
4. Trace imports and dependencies
5. Find ALL test files for the relevant code
6. Check docs/ for documentation of the feature area
7. Look for TODOs, FIXMEs referencing this issue

Output: ## Relevant Code, ## Type System & Interfaces, ## Call Chains, ## Test Coverage, ## Documentation, ## Constraints Found — each with citations.`;

const COMPETITIVE_ANALYSIS_PROMPT = `You are researching how competing agent frameworks implement a capability described in a GitHub issue.

Look at SOURCE CODE only. Do not fetch docs pages, blog posts, or tutorials.

Tools:
  node ${WEB_SEARCH_SCRIPT} gh-search-code "keyword" --repo owner/repo --max 10
  node ${WEB_SEARCH_SCRIPT} gh-file owner/repo path/to/file.py

Frameworks (only these three):
1. LangGraph — langchain-ai/langgraph (Python, libs/langgraph/langgraph/)
2. CrewAI — crewAIInc/crewAI (Python, src/crewai/)
3. AutoGen — microsoft/autogen (Python, autogen/)

For each: search for relevant code, fetch 2-3 source files, read the implementation. Note if they don't support it. Do NOT guess at APIs.

Output per framework: Supports? Implementation (cite paths). Code snippets. Gaps. Then: common patterns and what Pydantic AI should consider.`;

const PLAN_GENERATION_PROMPT = `You are a senior software engineer writing an implementation plan for a GitHub issue in Pydantic AI.

CRITICAL RULES:
- EVERY claim must have a citation: file path + line number, GitHub URL, or doc URL
- No filler. No "this will improve the developer experience." Just facts and steps.
- Call out what you are NOT sure about

Output: ## Goal, ## Prior Art & Competitive Landscape, ## Approach, ## Implementation Steps (with file paths and code snippets), ## Files to Modify, ## New Files, ## Test Plan, ## Documentation Changes, ## Risks and Pitfalls, ## Open Questions, ## References.`;

const REVIEW_PROMPT = `You are reviewing an implementation plan for Pydantic AI. Find gaps, hallucinated facts, wrong assumptions, missed edge cases. Verify citations. Do not praise — only identify problems.

For each issue: WHAT is wrong, WHERE in the plan, WHY it matters (cite evidence), HOW to fix it.

End with: SATISFIED (minor nits only) or NOT SATISFIED (material issues found).`;

const CONSOLIDATION_PROMPT = `Produce the final consolidated implementation plan from two reviewed plans. Take strongest elements from each. Every claim must have a citation. Zero filler.

Use format: ## Goal, ## Prior Art, ## Approach, ## Implementation Steps, ## Files, ## Tests, ## Docs, ## Risks, ## Open Questions, ## References.`;

// ─── Main Pipeline ───────────────────────────────────────────────────────────

async function main() {
	log(`Plan debate: issue #${ISSUE_NUMBER} in ${REPO}`);
	log(`Model A: ${MODEL_A}`);
	log(`Model B: ${MODEL_B}`);
	log(`Working directory: ${CWD}`);
	log("");

	// Phase 1: Issue Graph
	log("═══ PHASE 1/5: Crawling issue graph ═══");
	const { root, linked, fullContext } = await crawlIssueGraph(ISSUE_NUMBER);
	log(`Issue graph: ${1 + linked.length} items`);
	for (const item of linked) log(`  - ${item.type} #${item.number}: ${item.title}`);
	log("");

	// Phase 2: Research
	log("═══ PHASE 2/5: Deep parallel research (4 subagents) ═══");
	const cap = `${root.title}\n\n${root.body}`;

	const researchResults = await runConcurrent([
		{ label: `${MODEL_A} — codebase`, model: MODEL_A, task: `${CODEBASE_RESEARCH_PROMPT}\n\n${fullContext}` },
		{ label: `${MODEL_B} — codebase`, model: MODEL_B, task: `${CODEBASE_RESEARCH_PROMPT}\n\n${fullContext}` },
		{ label: `${MODEL_A} — competitive`, model: MODEL_A, task: `${COMPETITIVE_ANALYSIS_PROMPT}\n\n## Issue\n\n${cap}\n\n## Full Context\n\n${fullContext}` },
		{ label: `${MODEL_B} — competitive`, model: MODEL_B, task: `${COMPETITIVE_ANALYSIS_PROMPT}\n\n## Issue\n\n${cap}\n\n## Full Context\n\n${fullContext}` },
	]);

	const researchA = `## Codebase Research\n\n${researchResults[0].output}\n\n## Competitive Analysis\n\n${researchResults[2].output}`;
	const researchB = `## Codebase Research\n\n${researchResults[1].output}\n\n## Competitive Analysis\n\n${researchResults[3].output}`;
	log("");

	// Phase 3: Plans
	log("═══ PHASE 3/5: Generating independent plans ═══");
	const [planA, planB] = await Promise.all([
		llmComplete(MODEL_A, PLAN_GENERATION_PROMPT, `${fullContext}\n\n## Research Findings\n\n${researchA}`),
		llmComplete(MODEL_B, PLAN_GENERATION_PROMPT, `${fullContext}\n\n## Research Findings\n\n${researchB}`),
	]);
	log(`  ${MODEL_A}: ${(planA.length / 1024).toFixed(1)}KB`);
	log(`  ${MODEL_B}: ${(planB.length / 1024).toFixed(1)}KB`);
	log("");

	let currentPlanA = planA;
	let currentPlanB = planB;

	// Phase 4: Debate
	log("═══ PHASE 4/5: Debate ═══");
	for (let round = 1; round <= MAX_DEBATE_ROUNDS; round++) {
		log(`Round ${round}/${MAX_DEBATE_ROUNDS}: cross-reviewing...`);

		const [reviewA, reviewB] = await Promise.all([
			llmComplete(MODEL_A, REVIEW_PROMPT, `## Issue Graph\n${fullContext}\n\n## Plan Under Review (by ${MODEL_B})\n\n${currentPlanB}\n\n## Your Research\n\n${researchA}`),
			llmComplete(MODEL_B, REVIEW_PROMPT, `## Issue Graph\n${fullContext}\n\n## Plan Under Review (by ${MODEL_A})\n\n${currentPlanA}\n\n## Your Research\n\n${researchB}`),
		]);

		const satA = !reviewA.match(/\bnot\s+satisfied\b/gi) && !!reviewA.match(/\bsatisfied\b/gi);
		const satB = !reviewB.match(/\bnot\s+satisfied\b/gi) && !!reviewB.match(/\bsatisfied\b/gi);

		log(`  ${MODEL_A} ${satA ? "✅ satisfied" : "❌ not satisfied"} | ${MODEL_B} ${satB ? "✅ satisfied" : "❌ not satisfied"}`);

		if (satA && satB) { log("  Both satisfied!"); break; }

		log(`  Revising...`);
		const [revA, revB] = await Promise.all([
			llmComplete(MODEL_A, PLAN_GENERATION_PROMPT, `## Your Previous Plan\n\n${currentPlanA}\n\n## Review From ${MODEL_B}\n\n${reviewB}\n\n## Issue Graph\n${fullContext}\n\n## Your Research\n\n${researchA}\n\nRevise addressing the feedback.`),
			llmComplete(MODEL_B, PLAN_GENERATION_PROMPT, `## Your Previous Plan\n\n${currentPlanB}\n\n## Review From ${MODEL_A}\n\n${reviewA}\n\n## Issue Graph\n${fullContext}\n\n## Your Research\n\n${researchB}\n\nRevise addressing the feedback.`),
		]);
		currentPlanA = revA;
		currentPlanB = revB;
		log(`  Revised: ${MODEL_A} (${(revA.length / 1024).toFixed(1)}KB) | ${MODEL_B} (${(revB.length / 1024).toFixed(1)}KB)`);
	}
	log("");

	// Phase 5: Consolidation
	log("═══ PHASE 5/5: Consolidating ═══");
	const consolidated = await llmComplete(
		MODEL_A, CONSOLIDATION_PROMPT,
		`## Issue Graph\n${fullContext}\n\n## ${MODEL_A}'s Plan\n\n${currentPlanA}\n\n## ${MODEL_B}'s Plan\n\n${currentPlanB}\n\n## ${MODEL_A}'s Research\n\n${researchA}\n\n## ${MODEL_B}'s Research\n\n${researchB}`,
	);

	const linkedSummary = linked.length > 0
		? linked.map((i) => `- ${i.type} #${i.number}: ${i.title} (${i.url})`).join("\n")
		: "None found.";

	const plan = `# Implementation Plan: Issue #${root.number}

> **${root.title}**
> ${root.url}

> Generated via dual-model debate (${MODEL_A} + ${MODEL_B})
> Date: ${new Date().toISOString().split("T")[0]}
> Issue graph: ${1 + linked.length} items crawled

## Linked Issues & PRs

${linkedSummary}

${consolidated}

---

## Appendix

<details>
<summary>${MODEL_A}'s Final Plan</summary>

${currentPlanA}

</details>

<details>
<summary>${MODEL_B}'s Final Plan</summary>

${currentPlanB}

</details>

<details>
<summary>${MODEL_A}'s Research</summary>

${researchA}

</details>

<details>
<summary>${MODEL_B}'s Research</summary>

${researchB}

</details>`;

	// Write plan
	const planPath = path.join(CWD, "PLAN.md");
	fs.writeFileSync(planPath, plan, "utf-8");
	log(`Plan written to ${planPath} (${(plan.length / 1024).toFixed(1)}KB)`);

	if (SKIP_PR) {
		log("Skipping PR creation (--skip-pr)");
		return;
	}

	// Create PR
	log("Creating PR...");
	const branch = `plan/issue-${ISSUE_NUMBER}`;

	for (const cmd of [
		`git checkout -b ${branch}`,
		`git add PLAN.md`,
		`git commit -m "Add implementation plan for issue #${ISSUE_NUMBER}"`,
		`git push -u origin ${branch}`,
	]) {
		const r = await exec(cmd);
		if (r.code !== 0) { log(`FAILED: ${cmd}\n${r.stderr}`); process.exit(1); }
	}

	const prBody = `Implementation plan generated via dual-model debate (${MODEL_A} + ${MODEL_B}).\n\nCloses #${ISSUE_NUMBER}`;
	const prResult = await exec(
		`gh pr create --draft --title "Plan: Issue #${ISSUE_NUMBER}" --body "${prBody.replace(/"/g, '\\"')}" --head ${branch} --repo ${REPO}`,
	);
	if (prResult.code !== 0) {
		log(`PR creation failed: ${prResult.stderr}`);
		process.exit(1);
	}
	log(`✅ PR created: ${prResult.stdout.trim()}`);
}

main().catch((err) => {
	console.error("Fatal:", err.message);
	process.exit(1);
});
