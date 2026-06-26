from __future__ import annotations

from pathlib import Path


def test_localstack_ci_does_not_start_account_gated_service() -> None:
    workflow = Path(__file__).parents[3] / '.github' / 'workflows' / 'main.yml'
    lines = workflow.read_text().splitlines()

    action_index = lines.index(
        '      - uses: LocalStack/setup-localstack@7c8a0cb3405bc58be4c8f763f812aa000bc46303 # v0.3.2'
    )
    action_block = lines[action_index : action_index + 4]

    assert "          skip-startup: 'true'" in action_block


def test_localstack_ci_does_not_advertise_an_external_endpoint() -> None:
    workflow = Path(__file__).parents[3] / '.github' / 'workflows' / 'main.yml'
    lines = workflow.read_text().splitlines()

    assert not any('LOCALSTACK_ENDPOINT_URL:' in line for line in lines)


def test_localstack_ci_acknowledges_community_account_requirement() -> None:
    workflow = Path(__file__).parents[3] / '.github' / 'workflows' / 'main.yml'
    lines = workflow.read_text().splitlines()

    assert "      LOCALSTACK_ACKNOWLEDGE_ACCOUNT_REQUIREMENT: '1'" in lines
