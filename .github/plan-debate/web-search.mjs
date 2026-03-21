#!/usr/bin/env node
/**
 * Web search helper for plan-debate subagents.
 *
 * Usage:
 *   node web-search.mjs search "langgraph agent tools"
 *   node web-search.mjs fetch "https://docs.example.com/page"
 *   node web-search.mjs gh-search-code "class Agent" --repo langchain-ai/langgraph
 *   node web-search.mjs gh-search-repos "agentic framework"
 *   node web-search.mjs gh-readme langchain-ai/langgraph
 *   node web-search.mjs gh-file langchain-ai/langgraph path/to/file.py
 */

import { execFileSync } from "node:child_process";

const MAX_CONTENT = 15000;
const args = process.argv.slice(2);
const command = args[0];

if (!command) {
	console.log(`Usage:
  node web-search.mjs search <query> [--max N]
  node web-search.mjs fetch <url>
  node web-search.mjs gh-search-code <query> --repo <owner/repo> [--max N]
  node web-search.mjs gh-search-repos <query> [--max N]
  node web-search.mjs gh-readme <owner/repo>
  node web-search.mjs gh-file <owner/repo> <path>`);
	process.exit(0);
}

// ─── Arg Parsing ─────────────────────────────────────────────────────────

function getFlag(name) {
	const idx = args.indexOf(name);
	return idx !== -1 && args[idx + 1] ? args[idx + 1] : null;
}

// Extract positional args (skip flags and their values)
const positional = [];
for (let i = 1; i < args.length; i++) {
	if (args[i].startsWith("--")) { i++; } else { positional.push(args[i]); }
}
const query = positional.join(" ");
const max = parseInt(getFlag("--max") || "10", 10);

// ─── Helpers ─────────────────────────────────────────────────────────────

function gh(ghArgs) {
	try {
		return execFileSync("gh", ghArgs, { encoding: "utf-8", timeout: 30000 }).trim();
	} catch {
		return "";
	}
}

function truncate(text) {
	return text.length > MAX_CONTENT ? text.slice(0, MAX_CONTENT) + "\n\n[TRUNCATED]" : text;
}

function stripHtml(html) {
	return html
		.replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
		.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
		.replace(/<nav[^>]*>[\s\S]*?<\/nav>/gi, "")
		.replace(/<footer[^>]*>[\s\S]*?<\/footer>/gi, "")
		.replace(/<[^>]+>/g, " ")
		.replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/&lt;/g, "<")
		.replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'")
		.replace(/\s+/g, " ")
		.trim();
}

function extractText(html) {
	const mainMatch = html.match(/<main[^>]*>([\s\S]*?)<\/main>/i)
		|| html.match(/<article[^>]*>([\s\S]*?)<\/article>/i)
		|| html.match(/<body[^>]*>([\s\S]*?)<\/body>/i);

	let text = (mainMatch ? mainMatch[1] : html)
		.replace(/<h([1-6])[^>]*>([\s\S]*?)<\/h\1>/gi, "\n\n## $2\n\n")
		.replace(/<li[^>]*>([\s\S]*?)<\/li>/gi, "\n- $1")
		.replace(/<pre[^>]*>([\s\S]*?)<\/pre>/gi, "\n```\n$1\n```\n")
		.replace(/<code[^>]*>([\s\S]*?)<\/code>/gi, "`$1`")
		.replace(/<br\s*\/?>/gi, "\n")
		.replace(/<p[^>]*>/gi, "\n\n");

	return truncate(stripHtml(text).replace(/\n{3,}/g, "\n\n").trim());
}

function json(obj) { console.log(JSON.stringify(obj, null, 2)); }

// ─── Commands ─────────────────────────────────────────────────────────────

async function searchWeb() {
	// GitHub repo search — reliable, no API key needed
	const result = gh(["search", "repos", query, "--json", "name,url,description,stargazersCount", "--limit", String(max)]);
	if (result) {
		try {
			const repos = JSON.parse(result);
			json({ query, resultCount: repos.length, results: repos.map((r) => ({ url: r.url, title: r.name, snippet: r.description || "", stars: r.stargazersCount })) });
			return;
		} catch { /* fall through */ }
	}
	json({ query, resultCount: 0, results: [] });
}

async function fetchUrl() {
	const url = args[1];
	try {
		const resp = await fetch(url, {
			headers: { "User-Agent": "Mozilla/5.0", "Accept": "text/html,application/json,text/plain" },
			signal: AbortSignal.timeout(15000),
			redirect: "follow",
		});
		if (!resp.ok) { json({ url, error: `HTTP ${resp.status}`, content: "" }); return; }

		const ct = resp.headers.get("content-type") || "";
		const body = await resp.text();

		let content;
		if (ct.includes("json")) {
			try { content = truncate(JSON.stringify(JSON.parse(body), null, 2)); } catch { content = truncate(body); }
		} else if (ct.includes("html")) {
			content = extractText(body);
		} else {
			content = truncate(body);
		}
		json({ url, contentLength: body.length, content });
	} catch (e) {
		json({ url, error: e.message, content: "" });
	}
}

function ghSearchCode() {
	const repo = getFlag("--repo");
	if (!repo) { json({ error: "Missing --repo" }); return; }
	const result = gh(["search", "code", query, "--repo", repo, "--json", "path,textMatches", "--limit", String(max)]);
	console.log(result || JSON.stringify({ query, repo, results: [] }));
}

function ghSearchRepos() {
	const result = gh(["search", "repos", query, "--json", "name,url,description,stargazersCount", "--limit", String(max)]);
	console.log(result || JSON.stringify({ query, results: [] }));
}

function ghReadme() {
	const result = gh(["api", `repos/${args[1]}/readme`, "--jq", ".content"]);
	if (result) {
		const decoded = Buffer.from(result, "base64").toString("utf-8");
		json({ repo: args[1], content: truncate(decoded) });
	} else {
		json({ repo: args[1], error: "Could not fetch README", content: "" });
	}
}

function ghFile() {
	const result = gh(["api", `repos/${args[1]}/contents/${args[2]}`, "--jq", ".content"]);
	if (result) {
		const decoded = Buffer.from(result, "base64").toString("utf-8");
		json({ repo: args[1], path: args[2], content: truncate(decoded) });
	} else {
		json({ repo: args[1], path: args[2], error: "Could not fetch file", content: "" });
	}
}

// ─── Main ─────────────────────────────────────────────────────────────────

switch (command) {
	case "search": await searchWeb(); break;
	case "fetch": await fetchUrl(); break;
	case "gh-search-code": ghSearchCode(); break;
	case "gh-search-repos": ghSearchRepos(); break;
	case "gh-readme": ghReadme(); break;
	case "gh-file": ghFile(); break;
	default: console.log(`Unknown command: ${command}`); process.exit(1);
}
