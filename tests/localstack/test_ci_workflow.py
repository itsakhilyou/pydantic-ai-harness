from __future__ import annotations

from pathlib import Path


def _workflow_lines() -> list[str]:
    workflow = Path(__file__).parents[2] / '.github' / 'workflows' / 'main.yml'
    return workflow.read_text().splitlines()


def test_localstack_ci_lets_the_capability_manage_the_container() -> None:
    lines = _workflow_lines()
    action_index = lines.index(
        '      - uses: LocalStack/setup-localstack@7c8a0cb3405bc58be4c8f763f812aa000bc46303 # v0.3.2'
    )
    action_block = lines[action_index : action_index + 4]

    assert "          skip-startup: 'true'" in action_block


def test_localstack_ci_does_not_advertise_an_external_endpoint() -> None:
    lines = _workflow_lines()

    assert not any('LOCALSTACK_ENDPOINT_URL:' in line for line in lines)


def test_localstack_ci_authenticates_with_an_auth_token() -> None:
    lines = _workflow_lines()

    # The single image requires a token since LocalStack 2026.03.0; the deprecated
    # LOCALSTACK_ACKNOWLEDGE_ACCOUNT_REQUIREMENT bypass expired and must not return.
    assert '      LOCALSTACK_AUTH_TOKEN: ${{ secrets.LOCALSTACK_AUTH_TOKEN }}' in lines
    assert "      LOCALSTACK_REQUIRE_AUTH_TOKEN: '1'" in lines
    assert not any('LOCALSTACK_ACKNOWLEDGE_ACCOUNT_REQUIREMENT' in line for line in lines)
