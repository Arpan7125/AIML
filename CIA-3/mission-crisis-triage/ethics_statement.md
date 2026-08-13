# Ethics & Responsible-Use Statement — Disaster Message Triage Model

## Bias & fairness
The training data over-represents specific disasters — heavily the 2010 Haiti earthquake —
and specific languages/regions. The SHAP explainability analysis in the notebook confirms
this concretely: the model's top Critical-class signals are literally *earthquake*, *quake*,
and *aftershocks*, meaning it has partly learned "this disaster" rather than "disasters in
general." A model trained on this corpus should **not** be assumed to generalise to a new
disaster type, language, or informal social-media slang without re-validation on
locally-representative messages first.

## Privacy
Real disaster messages can contain names, exact locations, and health details. This project
uses only the public, pre-released Figure Eight dataset. Any production deployment must strip
or hash direct identifiers before storage/logging, and restrict raw-message access to
authorised responders only.

## Uncertainty
The model outputs class probabilities, not certainties (see the live-demo cell in the
notebook, which prints a full probability distribution per class). Messages with a
close-to-even probability split between Critical and High should be routed to a human
reviewer rather than auto-classified — operational confidence thresholds should be tuned
separately from the macro-F1 optimisation used for model selection.

## False-negative vs. false-positive cost
In this application, a false negative — a genuinely Critical message mis-classified as
Low/High — is far more costly than a false positive, which merely wastes a reviewer's few
minutes. Accuracy alone hides this failure mode, so the notebook reports **Critical-class
recall** separately for every model and trains with class-weighting so the rare Critical
class isn't drowned out by the much larger Low class.

## Human oversight
This system is designed as **decision support** — a prioritised queue for human coordinators
— not an autonomous dispatcher. A trained human should remain the final decision-maker,
especially for anything the model scores as Critical or flags as low-confidence.

## Limits on deployment
Do not deploy this model as-is on a live disaster without, at minimum:
1. Piloting on a held-out sample of messages from the *specific* disaster/region/language in
   question, given the generalisation risk identified above.
2. A human-in-the-loop review step for all Critical/uncertain predictions.
3. A monitoring plan to catch performance drift as the vocabulary used to describe the
   disaster evolves over the following days/weeks.
4. A clear escalation path for messages the model is uncertain about.
