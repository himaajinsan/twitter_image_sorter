import argparse
from image_pipeline import analyze_folder, assign_clusters, caption_folder, search


def main():
    parser = argparse.ArgumentParser(description="ローカル画像タグ付け・特徴量・検索")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="CLIP特徴量をSQLiteへ保存")
    analyze.add_argument("folder")
    analyze.add_argument("--database", default="images.sqlite3")
    analyze.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    analyze.add_argument("--no-resume", action="store_true")
    cluster = sub.add_parser("cluster", help="CLIP特徴量をクラスタリング")
    cluster.add_argument("--database", default="images.sqlite3")
    cluster.add_argument("--clusters", type=int, default=20)
    caption = sub.add_parser("caption", help="Florence-2キャプションをSQLiteへ保存")
    caption.add_argument("folder")
    caption.add_argument("--database", default="images.sqlite3")
    caption.add_argument("--model", default="florence-community/Florence-2-base")
    caption.add_argument("--no-resume", action="store_true")
    find = sub.add_parser("search", help="タグ・キャプション・パスを自然言語検索")
    find.add_argument("query", nargs="+")
    find.add_argument("--database", default="images.sqlite3")
    find.add_argument("--limit", type=int, default=50)
    find.add_argument("--semantic", action="store_true",
                      help="保存済みCLIP特徴量との意味検索（モデル名も指定）")
    find.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    args = parser.parse_args()
    if args.command == "analyze":
        result = analyze_folder(args.folder, args.database, clip_model=args.clip_model,
                                resume=not args.no_resume,
                                progress=lambda i, total, done, errors, *_:
                                print(f"\r{i}/{total} (保存 {done}, エラー {errors})", end="", flush=True))
        print(f"\n完了: 保存 {result[0]}, エラー {result[1]}")
    elif args.command == "cluster":
        print(f"{assign_clusters(args.database, clusters=args.clusters)} クラスタを作成しました")
    elif args.command == "caption":
        result = caption_folder(
            args.folder, args.database, model_name=args.model, resume=not args.no_resume,
            progress=lambda i, total, done, errors, *_:
            print(f"\r{i}/{total} (保存 {done}, エラー {errors})", end="", flush=True),
        )
        print(f"\n完了: 保存 {result[0]}, エラー {result[1]}")
    else:
        model = args.clip_model if args.semantic else None
        for row in search(args.database, " ".join(args.query), limit=args.limit, clip_model=model):
            print(f"[score={row['confidence']:.3f} cluster={row['cluster_id']}] {row['path']}")


if __name__ == "__main__":
    main()
