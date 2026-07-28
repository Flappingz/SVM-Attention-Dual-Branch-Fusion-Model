# Input data format

The preparation pipeline expects an authorized local export arranged by study
group and user. The repository does not contain real social-media records; the
layout below defines the interface for governed or independently collected data.

## Directory layout

```text
<raw-root>/
├── control/
│   ├── users.csv
│   └── <user-id>/
│       ├── <user-id>.csv
│       ├── img/原创微博图片/
│       └── live_photo/原创微博Live Photo视频/
└── self_reporting_ocd/
    ├── users.csv
    └── <user-id>/
        ├── <user-id>.csv
        ├── img/原创微博图片/
        └── live_photo/原创微博Live Photo视频/
```

Group directory names and labels are configured in
`configs/study.keyword-post-removed.example.json`; they are not inferred from the
data. An alternative static-image directory named `image/原创微博图片/` is also
recognized.

## User table

Each group contains a UTF-8 or UTF-8-with-BOM `users.csv`. The required column is:

| Column | Meaning |
| --- | --- |
| `用户id` | Platform user identifier; it must match the corresponding directory name |

Additional profile columns may be present, but the current preparation path does
not copy them into model inputs.

## Post table

Each user directory contains `<user-id>.csv`. The preparation code reads the
following columns:

| Column | Requirement | Use |
| --- | --- | --- |
| `id` | Required | Post identifier and media linkage |
| `正文` | Optional | Post text |
| `位置` | Optional | Location text appended during cleaning |
| `完整日期` | Preferred | Post timestamp |
| `日期` | Fallback | Timestamp when `完整日期` is absent |
| `点赞数` | Optional | Nonnegative interaction count |
| `评论数` | Optional | Nonnegative interaction count |
| `转发数` | Optional | Nonnegative interaction count |

Rows without a post ID are discarded. Duplicate post IDs are deduplicated by the
last encountered row and are reported by `audit-raw`. Invalid or absent counts
are handled by the metadata parser; malformed dates are reported by the audit.

## Media linkage

Static images may be stored under either recognized image directory. Live Photo
videos use `live_photo/原创微博Live Photo视频/`. The filename stem must contain the
numeric post ID as its second underscore-separated field, for example:

```text
prefix_123456789_1.jpg
prefix_123456789_1.mp4
```

The optional final numeric field is treated as the media ordinal. Files that
cannot be linked to a post are counted by `audit-raw` and remain governed raw
material; they are not silently converted into linked model inputs.

## Validation and pseudonymization

Before preparation, run:

```powershell
python -m ocd_v3 --config configs/study.keyword-post-removed.example.json audit-raw
```

Review the emitted inventory counts and errors before running `prepare`.
Preparation replaces raw user, post, and media identifiers with stable keyed
HMAC-SHA256 pseudonyms. The key and identity map must remain outside the Git
worktree and outside the general artifact root. See
[`data_governance.md`](data_governance.md) for the storage boundary.

The synthetic constructors in `tests/helpers.py` illustrate the minimum schema
used by automated tests; they do not represent or include study participants.