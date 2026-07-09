"""Back an agent's progressively-disclosed skills with a Logfire-managed variable."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from logfire.variables import Variable
from pydantic import BaseModel, Field
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability
from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness.logfire._managed_variable import ManagedVariableCapability

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions
    from pydantic_ai.capabilities.abstract import CapabilityDescription

# There is no first-party Logfire "skill management" feature reserving this prefix, so `skill__` is a
# harness convention: it namespaces the backing managed variable and keeps it visually grouped with
# the other managed-capability variables. One variable holds the whole list of skills for an agent.
_SKILL_VARIABLE_PREFIX = 'skill__'


class ManagedSkill(BaseModel):
    """One skill in a [`ManagedSkills`][pydantic_ai_harness.logfire.ManagedSkills] list.

    A skill is **instructions only** -- a named, described bundle of guidance the model can pull in
    on demand. It carries no tool code: nothing executable is ever downloaded from Logfire, which is
    what makes managed skills safe to steer from the UI. The `description` is the catalog blurb the
    model routes on; the `instructions` are revealed only after the model loads the skill.
    """

    name: str = Field(min_length=1)
    """Stable identifier for the skill, unique within the list. Used as the capability `id` the model
    references when it loads the skill, so keep it stable across edits."""

    description: str
    """Short blurb shown in the load catalog, so the model can decide whether to load the skill.
    Keep it to what the skill is *for*; the detailed guidance goes in `instructions`."""

    instructions: str
    """The guidance revealed once the model loads the skill -- added to the system prompt only after
    the model calls the `load_capability` tool for it."""


@dataclass
class _Skill(AbstractCapability[AgentDepsT]):
    """One resolved [`ManagedSkill`][pydantic_ai_harness.logfire.ManagedSkill] as a deferred capability.

    Uses the framework's progressive-disclosure mechanism: with `defer_loading=True` its instructions
    stay hidden and it appears only as a catalog entry (keyed by its `id`, described by
    `get_description`) until the model calls the framework's `load_capability` tool for it, at which
    point `get_instructions` is revealed. No tool code is ever involved -- a skill contributes
    instructions and nothing else.
    """

    skill: ManagedSkill = field(compare=False)
    """The resolved skill this capability discloses."""

    def __post_init__(self) -> None:
        # A stable `id` (required for `defer_loading`) is the skill's own name, so message history and
        # the load catalog can identify it across the run.
        self.id = self.skill.name
        self.defer_loading = True

    def get_description(self) -> CapabilityDescription[AgentDepsT] | None:
        """The catalog blurb shown alongside the `load_capability` tool before the skill is loaded."""
        return self.skill.description or None

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """The guidance added to the system prompt, revealed only once the model loads the skill."""
        return self.skill.instructions or None


@dataclass
class ManagedSkills(ManagedVariableCapability[AgentDepsT, 'list[ManagedSkill]']):
    """Back an agent's progressively-disclosed skills with a Logfire-managed variable.

    Drop this capability onto any agent and you can publish a catalog of **skills** -- named,
    described bundles of instructions the model pulls in on demand -- from the Logfire UI, versioned,
    labelled, and rolled out, without redeploying. A name of `support_agent` resolves the variable
    `skill__support_agent`.

    ```python
    import logfire
    from pydantic_ai import Agent

    from pydantic_ai_harness.logfire import ManagedSkills

    logfire.configure()

    agent = Agent(
        'openai:gpt-5',
        capabilities=[ManagedSkills('support_agent', label='production')],
    )
    result = agent.run_sync('How do I request a refund?')
    ```

    **Instructions only -- safe by construction:** a [`ManagedSkill`][pydantic_ai_harness.logfire.ManagedSkill]
    carries a `name`, a `description`, and `instructions` -- and nothing executable. No tool code is
    ever downloaded from Logfire, so a managed skill can only ever add guidance, never run code. This
    is a deliberate limit: publishing runnable tools from a remote UI is unsafe, so skills stop at
    instructions.

    **Progressive disclosure:** each skill is a *deferred* capability
    ([`defer_loading`](https://ai.pydantic.dev/capabilities/)). The model first sees only a catalog
    -- each skill's `name` and `description` -- next to the framework's `load_capability` tool. Its
    `instructions` are added to the system prompt only after the model loads it, so a large skill
    library costs a short catalog rather than a bloated prompt. The catalog and loader tool are
    provided by the framework for any deferred capability; this capability just contributes the
    skills.

    The list is resolved **once per run**, inside
    [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run] (the deferred capabilities must
    be in place before the framework assembles the catalog), and the
    [`ResolvedVariable`][logfire.variables.ResolvedVariable] is kept open as a context manager for the
    whole run -- so the selected label and version ride as baggage on every child span of the agent
    run.

    **Fallback semantics:** with no skills published (or when the remote value can't be validated),
    the logfire SDK falls back to the code default (an empty list), so the run simply proceeds with no
    skills -- never a crashed run. Two skills sharing a `name` would collide on their capability `id`,
    so the last one wins and a warning is emitted rather than breaking the run.

    Pass an existing [`logfire.variables.Variable`][logfire.variables.Variable] as `name` instead of a
    skills name when you want to use a variable you defined yourself.
    """

    name: str | Variable[list[ManagedSkill]] | None = None
    """The managed skills name (declared as the variable `skill__<name>`), or a pre-built
    `logfire.Variable`. When omitted, the variable is derived from the agent's own `name` at run time
    (`skill__<agent name>`); the agent must then have a `name`."""

    default: list[ManagedSkill] | None = None
    """Code-default skill list. When omitted, an empty list (no skills) is used -- a sensible default
    meaning "no skills yet", which is also the auto-create story. Ignored when `name` is a
    `Variable`."""

    def __post_init__(self) -> None:
        self._setup_variable(
            self.name,
            prefix=_SKILL_VARIABLE_PREFIX,
            value_type=list[ManagedSkill],
            default=self.default if self.default is not None else [],
        )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Resolve the skill list and assemble the per-run deferred capabilities from it.

        Resolution happens here (not in `wrap_run`) because the deferred `_Skill` capabilities must be
        present before the framework assembles the load catalog. The returned `CombinedCapability`
        carries the base's baggage holder alongside one deferred `_Skill` per skill, so all of them
        flow through the run as siblings of the agent's own capabilities. An empty (or invalid,
        code-defaulted) list yields just the baggage holder, and the run proceeds with no skills.
        """
        resolved = self._resolve(ctx)
        children = self._materialize_skills(resolved.value)
        return CombinedCapability([self._resolved_holder(resolved), *children])

    def _materialize_skills(self, skills: list[ManagedSkill]) -> list[AbstractCapability[AgentDepsT]]:
        """Build one deferred `_Skill` per skill, de-duplicating by name (last wins, with a warning).

        A duplicate name would produce two capabilities with the same `id`, which the framework rejects
        as a collision; folding to a last-wins map (matching how the UI form is edited) keeps a bad
        managed value from breaking the run, while the warning signals that something upstream emitted
        a duplicate.
        """
        by_name: dict[str, ManagedSkill] = {}
        for skill in skills:
            if skill.name in by_name:
                warnings.warn(
                    f'Multiple managed skills are named {skill.name!r}; the last one wins.',
                    stacklevel=2,
                )
            by_name[skill.name] = skill
        return [_Skill[AgentDepsT](skill) for skill in by_name.values()]
