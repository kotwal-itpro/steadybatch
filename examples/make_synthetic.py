"""Make a synthetic customer-support dataset with known answers.

Every conversation is built from a template, so we know the right answer for
every field. That lets the benchmark measure accuracy as well as reliability,
with no private data involved.

    python examples/make_synthetic.py --n 1000 > examples/support_tickets.jsonl
"""

from __future__ import annotations

import argparse
import json
import random

PRODUCTS = ["router", "laptop", "phone", "smart thermostat", "wireless earbuds", "printer"]
ISSUES = {
    "billing": ["I was charged twice for my {p}.", "My invoice for the {p} shows the wrong amount."],
    "shipping": ["My {p} still hasn't arrived after two weeks.", "The {p} was delivered to the wrong address."],
    "defect": ["My {p} stopped working after three days.", "The {p} keeps restarting on its own."],
    "how_to": ["How do I reset my {p} to factory settings?", "Can you tell me how to pair the {p} with my account?"],
}
MOODS = {
    "negative": ["This is really frustrating.", "I'm very unhappy with this."],
    "neutral": ["Thanks.", "Let me know what you need from me."],
    "positive": ["Otherwise I love the product!", "You've always been helpful, thank you."],
}
AGENT = [
    "Thanks for reaching out. Let me look into that for you.",
    "Sorry about that. Could you confirm your order number?",
]
FOLLOW_UP = ["Can someone call me back tomorrow?", "Please email me when it's fixed."]


def make(i: int, rng: random.Random) -> dict:
    product = rng.choice(PRODUCTS)
    issue = rng.choice(sorted(ISSUES))
    mood = rng.choice(sorted(MOODS))
    follow = rng.random() < 0.4
    customer = rng.choice(ISSUES[issue]).format(p=product) + " " + rng.choice(MOODS[mood])
    lines = [f"Customer: {customer}", f"Agent: {rng.choice(AGENT)}", f"Customer: Order number is A{10000 + i}."]
    if follow:
        lines.append(f"Customer: {rng.choice(FOLLOW_UP)}")
    lines.append("Agent: Got it, I've noted that down.")
    return {
        "key": f"ticket-{i:07d}",
        "text": "Extract the fields from this conversation.\n\n" + "\n".join(lines),
        "labels": {"product": product, "issue_type": issue, "sentiment": mood, "needs_follow_up": follow},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    for i in range(args.n):
        print(json.dumps(make(i, rng)))


if __name__ == "__main__":
    main()
