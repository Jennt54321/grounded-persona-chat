#!/usr/bin/env python3
"""
Generate 100 diverse questions for RAG evaluation.
Stratified by dialogue (Apology, Meno, Gorgias, Republic).
Optionally validates each question has at least one retrievable passage.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Stratified question seeds (~25 per dialogue)
APOLOGY_QUESTIONS = [
    "What does Socrates say about the charges against him?",
    "How does Socrates respond to the accusation that he corrupts the youth?",
    "What is Socrates' defense regarding the charge of impiety?",
    "Why does Socrates claim the Oracle at Delphi said no one is wiser than him?",
    "What does Socrates say about death and whether it is good or evil?",
    "How does Socrates describe his mission to examine others?",
    "What does Socrates say about the unexamined life?",
    "Why does Socrates refuse to beg for mercy from the jury?",
    "What penalty does Socrates propose instead of death?",
    "How does Socrates respond to the death sentence?",
    "What does Socrates say to those who voted for his acquittal?",
    "What does Socrates say about his accusers Meletus and Anytus?",
    "How does Socrates explain his lack of fear of death?",
    "What does Socrates say about the soul after death?",
    "Why does Socrates say he would not stop philosophizing?",
    "What does Socrates say about the nature of wisdom?",
    "How does Socrates characterize his divine sign or daimonion?",
    "What does Socrates say about the value of virtue?",
    "How does Socrates describe his relationship with the city of Athens?",
    "What does Socrates say about his poverty and way of life?",
    "Why does Socrates say he is a gadfly to the city?",
    "What does Socrates say about the difficulty of escaping death?",
    "How does Socrates address the jury's verdict?",
    "What does Socrates say about the difficulty of avoiding injustice?",
    "What final words does Socrates offer about his fate?",
]

MENO_QUESTIONS = [
    "What is the first definition of virtue that Meno offers?",
    "How does Socrates respond to Meno's definition of virtue?",
    "What is Meno's paradox about inquiry?",
    "How does Socrates explain the theory of recollection?",
    "What does the slave boy demonstration show about learning?",
    "Does Socrates think virtue can be taught?",
    "What does Socrates say about knowledge and true belief?",
    "How does Socrates define virtue in the dialogue?",
    "What role does Anytus play in the Meno?",
    "What does Socrates say about the relationship between virtue and knowledge?",
    "How does Meno define virtue initially?",
    "What does Socrates say about the immortality of the soul?",
    "What is the geometric proof used in the slave boy example?",
    "Does Socrates think there are teachers of virtue?",
    "What does Socrates say about the difference between knowledge and opinion?",
    "How does the dialogue end regarding the nature of virtue?",
    "What does Socrates say about the role of divine dispensation in virtue?",
    "What is the method of hypothesis that Socrates uses?",
    "How does Socrates refute Meno's definitions of virtue?",
    "What does Socrates say about the desire for good things?",
    "How does the slave boy come to know the geometric answer?",
    "What does Socrates say about the teachability of virtue?",
    "What is the connection between virtue and wisdom in the Meno?",
    "How does Socrates respond when Meno claims he cannot inquire?",
    "What does Socrates say about the origins of virtue?",
    "What is the role of recollection in learning according to Socrates?",
]

GORGIAS_QUESTIONS = [
    "What is Gorgias' view of rhetoric?",
    "How does Socrates define rhetoric in the Gorgias?",
    "What does Socrates say about the difference between rhetoric and philosophy?",
    "What is Polus's view on whether it is better to do or suffer injustice?",
    "How does Socrates argue that doing injustice is worse than suffering it?",
    "What does Callicles say about nature and convention?",
    "How does Socrates respond to Callicles' view that the strong should rule?",
    "What does Socrates say about the tyrant's power and happiness?",
    "What is the relationship between pleasure and the good according to Socrates?",
    "How does Socrates argue against Callicles' hedonism?",
    "What does Gorgias say rhetoric is the art of?",
    "What does Socrates say about flattery and true arts?",
    "How does Socrates distinguish between experience and art?",
    "What does Polus say about power and justice?",
    "What is Callicles' view of justice and natural right?",
    "How does Socrates compare the orator to the tyrant?",
    "What does Socrates say about the best life?",
    "How does Socrates use the myth at the end of the Gorgias?",
    "What does Socrates say about punishment and the soul?",
    "What is the role of Chaerephon in the Gorgias?",
    "How does Socrates argue that rhetoric is a form of flattery?",
    "What does Socrates say about the value of philosophy over rhetoric?",
    "How does Callicles defend the life of pleasure?",
    "What does Socrates say about the need for discipline in the soul?",
    "What is Socrates' final view on how one should live?",
    "How does Socrates refute Polus's claim about the happy tyrant?",
]

REPUBLIC_QUESTIONS = [
    "What is justice according to Socrates in the Republic?",
    "What is the analogy of the cave?",
    "What are the three parts of the soul according to Plato?",
    "What is the relationship between the soul and the city?",
    "What are the three waves of paradox in Book 5?",
    "What does Socrates say about the philosopher-king?",
    "What is the Form of the Good?",
    "How does Socrates define justice in the individual?",
    "What is the tripartite structure of the ideal city?",
    "What does Glaucon say about the ring of Gyges?",
    "What is the myth of Er?",
    "How does Socrates describe the education of the guardians?",
    "What does Socrates say about poetry and imitation?",
    "What is the distinction between the visible and intelligible realms?",
    "What are the four virtues of the city and soul?",
    "How does Socrates argue that the just life is happier?",
    "What does Thrasymachus say about justice?",
    "What is the theory of forms in the Republic?",
    "What does Socrates say about the degeneration of regimes?",
    "How does Socrates describe the tyrant's soul?",
    "What is the divided line analogy?",
    "What does Socrates say about women and the guardian class?",
    "How does Adeimantus contribute to the discussion of justice?",
    "What is the role of music and gymnastics in education?",
    "What does Socrates say about the unity of the virtuous soul?",
    "How does the Republic address the immortality of the soul?",
]

ALL_QUESTIONS = (
    [(q, "apology") for q in APOLOGY_QUESTIONS]
    + [(q, "meno") for q in MENO_QUESTIONS]
    + [(q, "gorgias") for q in GORGIAS_QUESTIONS]
    + [(q, "republic") for q in REPUBLIC_QUESTIONS]
)


def generate_questions(
    count: int = 100,
    validate: bool = False,
    books_dir: Path | None = None,
) -> list[dict]:
    """Generate count questions, optionally validating retrieval."""
    from src.retriever import Retriever

    questions: list[dict] = []
    seen: set[str] = set()
    retriever = Retriever(books_dir=books_dir or PROJECT_ROOT / "books") if validate else None

    for i, (q, book_hint) in enumerate(ALL_QUESTIONS):
        if len(questions) >= count:
            break
        q_trimmed = q.strip()
        if q_trimmed in seen:
            continue
        if validate and retriever:
            chunks = retriever.search(q_trimmed, top_k=1)
            if not chunks:
                continue
        seen.add(q_trimmed)
        questions.append({
            "id": len(questions) + 1,
            "question": q_trimmed,
            "book_hint": book_hint,
        })

    return questions[:count]


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation questions")
    parser.add_argument("--output", "-o", type=Path, default=PROJECT_ROOT / "eval" / "questions.json")
    parser.add_argument("--count", "-n", type=int, default=100)
    parser.add_argument("--validate", action="store_true", help="Ensure each question has retrievable passages")
    parser.add_argument("--books-dir", type=Path, default=None)
    args = parser.parse_args()

    questions = generate_questions(
        count=args.count,
        validate=args.validate,
        books_dir=args.books_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(questions)} questions to {args.output}")


if __name__ == "__main__":
    main()
