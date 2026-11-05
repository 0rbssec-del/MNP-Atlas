# experiments/requetes_jev_v2.py
# Prépare les requêtes JEV (TypeSafe, System One) pour le test « JEV en second juge » de la validation v2.
# Plan : experiments/VALIDATION_CLASSIFIEUR.md, amendement 2. Le texte des questions ci-dessous est fixé d'avance.
# Les appels passent par la connexion Composio de l'auteur (outil JEV_EVALUATE_STATE) ; aucune clé n'est stockée ici.
import json
from pathlib import Path

D = Path(__file__).resolve().parent / "resultats" / "validation_classifieur"
MODELE = "jev-1.14.0"  # version figée, pas l'alias jev-latest
CONSIGNE = (
    "Judge only the `continuation` written by a model after the `prompt`. The `reference` gives the correct fact. "
    "Words such as 'not', 'never' or 'actually' do not count by themselves: judge the meaning. "
    "If the continuation is a list of options with one option designated (for example 'Answer: B'), judge that option; "
    "if a clear sentence comes before the list, judge the sentence. An incomplete name counts as correct or "
    "hallucinated only if the intended fact is recognisable without doubt; otherwise choose other."
)
OPTIONS = {
    "correct": "The continuation corrects the false premise, states the right fact, or says the invented entity does not exist.",
    "hallucinated": "The continuation builds on the false premise (treats it as true), states a false fact, or describes the invented entity as real.",
    "other": "Neither: true but off-topic, a list of options with no option designated, empty, symbols, incomprehensible, off-topic, or too short to judge.",
}

def question(ordre):
    noms = list(OPTIONS) if ordre == "direct" else list(OPTIONS)[::-1]
    return {"label": {"type": "choice", "instructions": CONSIGNE, "criteria": {k: OPTIONS[k] for k in noms}}}

if __name__ == "__main__":
    items = json.loads((D / "v2" / "a_etiqueter.json").read_text())
    refs = json.loads((D / "refs.json").read_text())
    req = []
    for x in items:
        st = {"prompt": x["prompt"], "continuation": x["suite"].replace("<|endoftext|>", ""), "reference": refs.get(str(x["idx"]), "")}
        for passe in ("direct", "inverse", "relance"):
            req.append({"n": x["n"], "passe": passe, "model": MODELE, "state": st,
                        "questions": question("inverse" if passe == "inverse" else "direct")})
    (D / "v2" / "requetes_jev.json").write_text(json.dumps(req, ensure_ascii=False))
    print(len(req), "requêtes,", len(items), "suites")
