#!/usr/bin/env python3
"""
Scrape ALL Egypt car listings from dubizzle (dubizzle.com.eg) through ReefAPI
and save them to a CSV..

"""
import argparse
import csv
import json
import os
import sys
import time

import requests 

URL = "https://api.reefapi.com/dubizzle/v1/cars/search"
REACHABLE = 10_000
PAGE_SIZE = 100

# Price band edges in EGP. Last band is open-ended.
EDGES = [0, 50_000, 100_000, 150_000, 200_000, 250_000, 300_000, 400_000,
         500_000, 600_000, 800_000, 1_000_000, 1_250_000, 1_500_000,
         2_000_000, 3_000_000, 5_000_000, 10_000_000, 1_000_000_000]

COLUMNS = [
    # bookkeeping
    "listing_id", "url", "listed_date",
    # the ~10 features
    "title", "make", "model", "version", "year", "kilometers", "price", "currency",
    "transmission", "fuel_type", "body_type", "exterior_color", "condition",
    "payment_options", "location", "price_type", "is_verified",
    # safety net: everything dubizzle sent as attributes
    "attributes_json",
]

ALIASES = {
    "make": ("brand", "make"),
    "model": ("model",),
    "version": ("version",),
    "year": ("year",),
    "kilometers": ("kilometers", "kilometres"),
    "transmission": ("transmission type", "transmission"),
    "fuel_type": ("fuel type", "fuel_type"),
    "body_type": ("body type", "body_type"),
    "exterior_color": ("color", "colour", "exterior_color"),
    "condition": ("condition",),
    "payment_options": ("payment options",),
}


class OutOfCredits(Exception):
    pass


class Client:
    def __init__(self, key, max_credits, language):
        self.key = key
        self.max_credits = max_credits
        self.language = language
        self.spent = 0
        self.session = requests.Session()

    def call(self, body):
        """POST one cars/search request. Returns the `data` dict or None if the slice failed."""
        if self.max_credits and self.spent >= self.max_credits:
            raise OutOfCredits(f"reached your --max-credits cap ({self.max_credits})")
        body = {"country": "eg", "language": self.language, "limit": PAGE_SIZE, **body}
        for attempt in range(1, 5):
            try:
                r = self.session.post(
                    URL, json=body, timeout=90,
                    headers={"x-api-key": self.key, "content-type": "application/json"},
                )
            except requests.RequestException as e:
                print(f"  network error ({e}); retry {attempt}/4", file=sys.stderr)
                time.sleep(2 * attempt)
                continue
            if r.status_code == 402:
                raise OutOfCredits("API answered 402 (payment/credits required)")
            if r.status_code == 429:
                time.sleep(3 * attempt)
                continue
            try:
                js = r.json()
            except ValueError:
                time.sleep(2 * attempt)
                continue
            if js.get("ok"):
                self.spent += 1  # cars/search costs 1 credit per successful call
                return js["data"]
            err = js.get("error") or {}
            msg = f"{err.get('code')}: {err.get('message')}"
            if "credit" in msg.lower() or "quota" in msg.lower() or "balance" in msg.lower():
                raise OutOfCredits(msg)
            print(f"  API error {msg}; retry {attempt}/4", file=sys.stderr)
            time.sleep(2 * attempt)
        print(f"  giving up on request {body}", file=sys.stderr)
        return None


def pick(row, names):
    for a in row.get("attributes") or []:
        k = str(a.get("key", "")).lower()
        l = str(a.get("label", "") or a.get("name", "")).lower()
        if k in names or l in names:
            return a.get("value")
    for n in names:
        v = row.get(n)
        if v not in (None, "", []) and not isinstance(v, (list, dict)):
            return v
    return None


def results_of(data):
    for k in ("listings", "results"):
        if isinstance(data.get(k), list):
            return data[k]
    for v in data.values():  # Egypt may name the list differently
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    return []


def flatten(row):
    loc = row.get("location")
    if isinstance(loc, list):  # Egypt: [Egypt, Cairo, New Cairo] -> "New Cairo, Cairo"
        names = [x.get("name") for x in loc if isinstance(x, dict) and x.get("name")]
        loc = ", ".join(reversed(names[1:] if len(names) > 1 else names))
    elif isinstance(loc, dict):
        loc = ", ".join(p for p in (loc.get("area"), loc.get("emirate")) if p)
    listed = row.get("listed_date")
    if not listed and isinstance(row.get("listed_at"), (int, float)):
        listed = time.strftime("%Y-%m-%d", time.gmtime(row["listed_at"]))
    out = {
        "listing_id": row.get("listing_id"),
        "url": row.get("url"),
        "listed_date": listed,
        "title": row.get("title"),
        "price": row.get("price"),
        "currency": row.get("currency"),
        "location": loc,
        "price_type": row.get("price_type"),
        "is_verified": row.get("is_verified"),
        "attributes_json": json.dumps(row.get("attributes") or [], ensure_ascii=False),
    }
    for col, names in ALIASES.items():
        out[col] = pick(row, names)
    return out


class Scraper:
    def __init__(self, client, out_path, ckpt_path, sort):
        self.c = client
        self.out_path = out_path
        self.ckpt_path = ckpt_path
        self.sort = sort
        self.seen = set()
        self.done = set()
        self.new_rows = 0

        if os.path.exists(out_path):
            with open(out_path, newline="", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    self.seen.add(str(r.get("listing_id")))
        if os.path.exists(ckpt_path):
            with open(ckpt_path) as f:
                self.done = {tuple(x) for x in json.load(f)}
        is_new = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
        self.fh = open(out_path, "a", newline="", encoding="utf-8-sig")
        self.w = csv.DictWriter(self.fh, fieldnames=COLUMNS, extrasaction="ignore")
        if is_new:
            self.w.writeheader()

    def save_rows(self, data):
        n = 0
        for row in results_of(data):
            lid = str(row.get("listing_id"))
            if lid in self.seen:
                continue
            self.seen.add(lid)
            self.w.writerow(flatten(row))
            n += 1
        self.fh.flush()
        self.new_rows += n
        return n

    def mark_done(self, lo, hi):
        self.done.add((lo, hi))
        with open(self.ckpt_path, "w") as f:
            json.dump(sorted(self.done), f)

    def crawl(self, lo, hi):
        if (lo, hi) in self.done:
            return
        body = {"sort": self.sort, "price_min": lo, "price_max": hi, "page": 1}
        data = self.c.call(body)
        if data is None:
            return
        total = data.get("total_results", 0)
        reachable = data.get("reachable_results") or min(total, REACHABLE)
        got = self.save_rows(data)
        print(f"[{lo:>10,} - {hi:>13,} EGP] total={total:>6,}  page 1: +{got}  "
              f"(credits used: {self.c.spent}, rows so far: {len(self.seen):,})")
        if total == 0:
            self.mark_done(lo, hi)
            return
        if total > reachable and hi > lo:
            mid = (lo + hi) // 2
            print(f"   -> more than {REACHABLE:,} results, splitting at {mid:,}")
            self.crawl(lo, mid)
            self.crawl(mid + 1, hi)
            return
        pages = -(-min(total, reachable) // PAGE_SIZE)
        page = 2
        while page <= pages and data.get("has_more", True):
            data = self.c.call({**body, "page": page})
            if data is None:
                return  # leave slice unfinished so a re-run retries it
            got = self.save_rows(data)
            print(f"   page {page}/{pages}: +{got}  (credits used: {self.c.spent}, rows so far: {len(self.seen):,})")
            page += 1
        self.mark_done(lo, hi)

    def run(self):
        for lo, nxt in zip(EDGES, EDGES[1:]):
            self.crawl(lo, nxt - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dubizzle_egypt_cars.csv")
    ap.add_argument("--max-credits", type=int, default=0,
                    help="stop after this many credits (0 = run until the API says you're out)")
    ap.add_argument("--language", default="en", choices=["en", "ar"])
    ap.add_argument("--sort", default="price_asc",
                    choices=["newest", "oldest", "price_asc", "price_desc"])
    ap.add_argument("--probe", action="store_true",
                    help="make ONE call (1 credit) and print the raw first row, then exit")
    args = ap.parse_args()

    key = os.environ.get("REEF_KEY")
    if not key:
        sys.exit("Set your key first:  export REEF_KEY=your_reefapi_key")

    client = Client(key, args.max_credits, args.language)

    if args.probe:
        data = client.call({"limit": 5, "sort": "newest"})
        if data:
            print("keys:", list(data.keys()))
            print(f"total_results={data.get('total_results')} rows={len(results_of(data))}")
            print(json.dumps(data, indent=2, ensure_ascii=False)[:3500])
        return

    s = Scraper(client, args.out, args.out + ".checkpoint.json", args.sort)
    try:
        s.run()
        print("\nFinished every price band.")
    except OutOfCredits as e:
        print(f"\nStopped: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by you.")
    finally:
        s.fh.close()
        print(f"Saved {len(s.seen):,} unique listings to {args.out} "
              f"({s.new_rows:,} new this run, {client.spent} credits this run).")


if __name__ == "__main__":
    main()