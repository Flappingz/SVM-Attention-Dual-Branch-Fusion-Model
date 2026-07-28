from __future__ import annotations

import csv
import json
import os
from pathlib import Path

USER_COLUMNS = [
    "用户id",
    "昵称",
    "性别",
    "生日",
    "所在地",
    "IP属地",
    "学习经历",
    "公司",
    "注册时间",
    "阳光信用",
    "微博数",
    "粉丝数",
    "关注数",
    "简介",
    "主页",
    "头像",
    "高清头像",
    "微博等级",
    "会员等级",
    "是否认证",
    "认证类型",
    "认证信息",
    "上次记录微博信息",
]

POST_COLUMNS = [
    "id",
    "bid",
    "正文",
    "头条文章url",
    "原始图片url",
    "视频url",
    "Live Photo视频url",
    "位置",
    "日期",
    "工具",
    "点赞数",
    "评论数",
    "转发数",
    "话题",
    "@用户",
    "完整日期",
    "是否编辑过",
    "编辑次数",
]


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_raw_group(
    raw_root: Path,
    group: str,
    users: list[str],
    directory_users: list[str] | None = None,
) -> None:
    group_dir = raw_root / group
    group_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        group_dir / "users.csv",
        USER_COLUMNS,
        [{"用户id": uid, "昵称": f"user-{uid}"} for uid in users],
    )
    for index, uid in enumerate(directory_users if directory_users is not None else users):
        sid = str(900000 + index)
        subject_dir = group_dir / uid
        write_csv(
            subject_dir / f"{uid}.csv",
            POST_COLUMNS,
            [
                {
                    "id": sid,
                    "正文": "我提到OCD但这不是临床标签",
                    "位置": "测试位置",
                    "日期": "2026-01-02",
                    "完整日期": "2026-01-02 03:04:05",
                    "点赞数": "4",
                    "评论数": "2",
                    "转发数": "1",
                }
            ],
        )
        image_dir = subject_dir / "img" / "原创微博图片"
        image_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / f"{uid}T_{sid}.jpg").write_bytes(b"not-a-real-image")


def write_study_config(path: Path, raw_root: Path, artifact_root: Path) -> None:
    os.environ.setdefault("OCD_PSEUDONYMIZATION_KEY", "synthetic-test-key-not-for-production")
    payload = {
        "schema_version": 2,
        "paths": {
            "raw_root": str(raw_root),
            "artifact_root": str(artifact_root),
            "private_mapping_root": str(path.parent / "private-mappings"),
        },
        "dataset": {
            "groups": [
                {"directory": "control", "label_name": "control", "label_id": 0},
                {
                    "directory": "self_reporting_ocd",
                    "label_name": "self_reported_ocd",
                    "label_id": 1,
                },
            ],
            "keywords": ["ocd", "强迫症"],
            "keyword_mask": "[MASK]",
            "post_selection": {
                "maximum_posts_per_subject": 64,
                "strategy": "most_recent",
            },
        },
        "evaluation": {
            "outer_folds": 5,
            "split_seed": 7,
            "validation_fraction_within_outer_train": 0.125,
            "classification_threshold": 0.5,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
