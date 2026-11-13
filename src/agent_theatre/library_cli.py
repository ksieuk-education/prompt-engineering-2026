"""Просмотр книжной полки, проверка поиска и разбор сохранённой постановки без LLM."""
import argparse
import json
from pathlib import Path

from .analysis import read_trace
from .evidence import benchmark, compare_runs, dossier
from .library import Library


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Полнотекстовая библиотека агентского театра")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="найти реплики и ремарки во всём корпусе")
    search.add_argument("query")
    search.add_argument("--book", choices=("garden", "inspector", "dowry"), default="")
    search.add_argument("--limit", type=int, default=3)
    view = commands.add_parser("open", help="прочитать фрагмент и соседние реплики в том же явлении")
    view.add_argument("ref")
    view.add_argument("--radius", type=int, default=2)
    evaluation = commands.add_parser("bench", help="сравнить два метода на редакторской разметке")
    evaluation.add_argument("--json", action="store_true", help="все запросы, адреса, ранги и метрики")
    report = commands.add_parser("dossier", help="сверить с книгами цитаты спектакля и пересчитать ближайший финал")
    report.add_argument("run", type=Path)
    pair = commands.add_parser("compare", help="сопоставить два спектакля на одном плане и зерне")
    pair.add_argument("control", type=Path, help="спектакль без библиотеки")
    pair.add_argument("with_books", type=Path, help="спектакль с библиотекой")
    args = parser.parse_args(argv)
    try:
        lib = Library()
        if args.command == "dossier":
            print(dossier(read_trace(args.run), lib), end="")
        elif args.command == "compare":
            comparison = compare_runs(read_trace(args.control), read_trace(args.with_books), lib)
            print(json.dumps(comparison, ensure_ascii=False, indent=2))
        elif args.command == "bench":
            data = benchmark(lib)
            if args.json:
                print(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                print(f"С ответом: {data['positive']}. Без ответа: {data['negative']}.")
                for name, m in data["scores"].items():
                    print(f"{name}: Hit@1 {m['top1']}, Hit@3 {m['top3']}, MRR@3 {m['mrr3']:.3f}, "
                          f"верных пустых выдач {m['empty']}.")
        else:
            hits = (lib.search(args.query, book=args.book, limit=args.limit) if args.command == "search"
                    else lib.context(args.ref, args.radius))
            if not hits:
                print("Совпадений не найдено.")
            for p in hits:
                print(f"{p.ref} · {lib.books[p.book].title} · действие {p.act} · {p.heading}\n"
                      f"{p.speaker}: {p.text}\n{lib.source(p)}\n")
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
