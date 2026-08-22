import pytest

from models import AppError, RouteDecision, clean_question
from router import normalize_route


def test_blank_question_is_rejected() -> None:
    """A question that is only whitespace must be rejected, not silently trimmed to ''."""
    with pytest.raises(AppError) as error:
        clean_question("   ")
    assert error.value.status_code == 400


def test_superhero_route_requires_a_name() -> None:
    """Selecting the superhero source with no name to look up is an invalid route."""
    with pytest.raises(AppError):
        normalize_route(RouteDecision(sources=["superhero_api"]))


def test_unsupported_route_combination_is_rejected() -> None:
    """Documents and web search are not allowed to be combined in one route."""
    with pytest.raises(AppError):
        normalize_route(RouteDecision(sources=["text_rag", "web"]))


def test_names_are_dropped_when_superhero_is_not_selected() -> None:
    """Stray hero names are ignored if superhero_api was not actually selected."""
    decision = normalize_route(
        RouteDecision(sources=["web"], superhero_names=["Batman"])
    )
    assert decision.superhero_names == []