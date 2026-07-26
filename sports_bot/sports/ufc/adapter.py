"""UFC adapter skeleton — wire to UFC-Predictor data loaders when porting."""

from __future__ import annotations

from sports_bot.core.confidence import attach_confidence
from sports_bot.models.compubox import CompuBoxLine, differential
from sports_bot.models.markov_styles import StyleProfile, blend_with_model, markov_win_prob
from sports_bot.sports.base import Matchup, ModelPick, SportAdapter


class UfcAdapter(SportAdapter):
    sport = "ufc"

    def upcoming_matchups(self) -> list[Matchup]:
        # TODO: port card fetch from C:\UFC-Predictor (ESPN / UFC.com / cache).
        return [
            Matchup(
                event="DEMO UFC Card",
                sport=self.sport,
                selection_a="Fighter A",
                selection_b="Fighter B",
                start_time="",
                meta={"demo": True},
            )
        ]

    def score_matchup(self, matchup: Matchup) -> ModelPick:
        # Demo blend: CompuBox diffs + Markov styles + flat prior.
        a_box = CompuBoxLine(matchup.selection_a, 45, 100, 0.4, 0.6, 0.25, 0.15, 12)
        b_box = CompuBoxLine(matchup.selection_b, 38, 95, 0.2, 0.55, 0.3, 0.15, 12)
        diffs = differential(a_box, b_box)

        a_style = StyleProfile(matchup.selection_a, {"striker": 0.5, "pressure": 0.3, "wrestler": 0.2})
        b_style = StyleProfile(matchup.selection_b, {"wrestler": 0.5, "grappler": 0.3, "counter": 0.2})
        markov_p = markov_win_prob(a_style, b_style)

        # Toy ML stand-in from striking differentials
        toy_model = 0.5 + 0.15 * diffs["acc_diff"] + 0.05 * diffs["volume_diff"]
        toy_model = min(0.85, max(0.15, toy_model))
        prob_a = blend_with_model(markov_p, toy_model, markov_weight=0.3)

        conf = attach_confidence(prob_a)
        pick_name = matchup.selection_a if prob_a >= 0.5 else matchup.selection_b
        pick_prob = prob_a if prob_a >= 0.5 else 1.0 - prob_a
        reasons = [
            f"CompuBox acc_diff={diffs['acc_diff']:+.2f} volume_diff={diffs['volume_diff']:+.2f}",
            f"Markov style P(A)={markov_p:.2f} blended to {prob_a:.2f}",
            f"Confidence={conf['confidence_label']}",
        ]
        return ModelPick(
            matchup=matchup,
            selection=pick_name,
            prob=pick_prob,
            features={"prob_a": prob_a, "compubox": diffs, "markov_p": markov_p},
            reasons=reasons,
        )
