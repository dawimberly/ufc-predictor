"""Fight table sort: color → edge↓ → model_prob↓ → fight name↑."""

from src.ufc_dashboard import _fight_table_sort_key


def test_color_priority_blue_before_green_before_yellow_before_red() -> None:
    rows = [
        {"tier": "red", "sort_edge": 0.20, "sort_prob": 0.90, "sort_fight": "A vs B"},
        {"tier": "yellow", "sort_edge": 0.10, "sort_prob": 0.80, "sort_fight": "C vs D"},
        {"tier": "green", "sort_edge": 0.05, "sort_prob": 0.70, "sort_fight": "E vs F"},
        {"tier": "blue", "sort_edge": 0.01, "sort_prob": 0.60, "sort_fight": "G vs H"},
    ]
    ordered = sorted(rows, key=_fight_table_sort_key)
    assert [r["tier"] for r in ordered] == ["blue", "green", "yellow", "red"]


def test_within_color_edge_then_prob_then_fight_name() -> None:
    rows = [
        {"tier": "green", "sort_edge": 0.08, "sort_prob": 0.70, "sort_fight": "Zed vs Ann", "pick": "Zed"},
        {"tier": "green", "sort_edge": 0.12, "sort_prob": 0.65, "sort_fight": "Bob vs Cat", "pick": "Bob"},
        {"tier": "green", "sort_edge": 0.12, "sort_prob": 0.80, "sort_fight": "Amy vs Dan", "pick": "Amy"},
        {"tier": "green", "sort_edge": 0.12, "sort_prob": 0.80, "sort_fight": "Cal vs Eve", "pick": "Cal"},
    ]
    ordered = sorted(rows, key=_fight_table_sort_key)
    assert [r["pick"] for r in ordered] == ["Amy", "Cal", "Bob", "Zed"]


def test_missing_edge_sorts_last_within_tier() -> None:
    rows = [
        {"tier": "red", "sort_edge": -999.0, "sort_prob": 0.90, "sort_fight": "No Odds vs X"},
        {"tier": "red", "sort_edge": -0.05, "sort_prob": 0.55, "sort_fight": "Neg Edge vs Y"},
    ]
    ordered = sorted(rows, key=_fight_table_sort_key)
    assert ordered[0]["sort_fight"].startswith("Neg Edge")
    assert ordered[1]["sort_fight"].startswith("No Odds")
