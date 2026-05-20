// scripts/notion-sync.js
// Syncs pages with Status == "Published" from Notion to src/content/writing/
// Run: node --env-file=.env scripts/notion-sync.js
// Dry run: node --env-file=.env scripts/notion-sync.js --dry-run

import { Client } from "@notionhq/client";
import { NotionToMarkdown } from "notion-to-md";
import { mkdir, writeFile } from "fs/promises";
import path from "path";

// ── Config ────────────────────────────────────────────────────────────────────
// Map your Notion property names here (case-sensitive, must match exactly)
const PROP = {
  title:       ["Name", "Title", "title"],
  status:      ["Status", "status"],
  type:        ["Type", "type"],            // "blog" | "reading" | "note"
  tags:        ["Tags", "tags"],            // single-select
  description: ["Description", "description"], // rich_text or text
};

const PUBLISHED_VALUE = "Published";  // exact value in your Status property
const OUT_DIR  = "src/content/writing";
const IMG_DIR  = "public/assets/images/writing";
// ─────────────────────────────────────────────────────────────────────────────

const isDryRun = process.argv.includes("--dry-run");

const notion = new Client({ auth: process.env.NOTION_TOKEN });
const n2m    = new NotionToMarkdown({ notionClient: notion });

// ── Global block transformers ─────────────────────────────────────────────────
n2m.setCustomTransformer("table_of_contents", async (_block) => {
  return "<!--TOC-->";  // replaced in post-processing after headings are known
});

// Callout: no custom transformer -- notion-to-md's default correctly wraps
// both the header and child blocks inside blockquote (>) markers.

// ── Property helpers ──────────────────────────────────────────────────────────
function getProp(props, names) {
  for (const name of names) {
    const p = props[name];
    if (!p) continue;
    switch (p.type) {
      case "title":        return p.title.map((t) => t.plain_text).join("") || null;
      case "rich_text":    return p.rich_text.map((t) => t.plain_text).join("") || null;
      case "select":       return p.select?.name ?? null;
      case "status":       return p.status?.name ?? null;
      case "multi_select": return p.multi_select.map((s) => s.name);
      case "date":         return p.date?.start ?? null;
      case "checkbox":     return p.checkbox;
      default:             return null;
    }
  }
  return null;
}

function toSlug(title) {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function toWritingSlug(title) {
  return toSlug(title);
}

function toChildWritingSlug(parentTitle, childTitle) {
  return [toSlug(parentTitle), toSlug(childTitle)].filter(Boolean).join("-");
}

function getChildPageTitle(block) {
  return String(block?.child_page?.title ?? block?.parent ?? "")
    .replace(/^#{1,6}\s+/, "")
    .trim() || "Untitled";
}

function yamlStr(val) {
  if (val === null || val === undefined) return '""';
  const s = String(val);
  return /[:#\[\]{},&*?|<>=!%@`]/.test(s) ? `"${s.replace(/"/g, '\\"')}"` : s;
}

// ── Post-processing ──────────────────────────────────────────────────────────
// Notion "toggle" blocks become <details><summary> which remark treats as raw
// HTML, so math and lists inside them never render. Convert to ## headings.
function fixToggles(md) {
  const pattern = /<details>\s*<summary>([\s\S]*?)<\/summary>([\s\S]*?)<\/details>/g;
  return md.replace(pattern, (_, summary, content) => {
    return `## ${summary.trim()}\n\n${content.trim()}`;
  });
}

// Generate TOC from headings and replace the <!--TOC--> placeholder.
function fixTOC(md) {
  if (!md.includes("<!--TOC-->")) return md;
  const headings = [];
  const seen = new Map();
  let inFence = false;

  for (const line of md.split("\n")) {
    if (/^```/.test(line.trim())) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;

    const m = line.match(/^(#{1,6})\s+(.+)/);
    if (m) {
      const level = m[1].length;
      const text  = m[2].trim();
      const base  = text.toLowerCase().replace(/[^a-z0-9\s-]/g, "").replace(/\s+/g, "-");
      const count = seen.get(base) ?? 0;
      seen.set(base, count + 1);
      const id = count === 0 ? base : `${base}-${count}`;
      headings.push({ level, text, id });
    }
  }
  const toc = headings
    .map(({ level, text, id }) => `${"  ".repeat(level - 1)}- [${text}](#${id})`)
    .join("\n");
  return md.replace("<!--TOC-->", toc).replaceAll("<!--TOC-->", "");
}

// Notion mention links render as [[N] full citation title](notion-url).
// Convert to [[N]](#ref-N) so they link to the local References section,
// and add <span id="ref-N"> anchors to each [N] entry in that section.
function fixCitations(md) {
  md = md.replace(
    /\[\[(\d+)\][^\]]*\]\(https:\/\/www\.notion\.so\/[^)]+\)/g,
    (_, n) => `[[${n}]](#ref-${n})`
  );
  md = md.replace(
    /^(\[(\d+)\] )/gm,
    (_, _full, n) => `<span id="ref-${n}"></span>[${n}] `
  );
  return md;
}

// Strip leading blank quote lines left by empty Notion blocks inside callouts/quotes.
function fixQuotes(md) {
  return md.replace(/^(> *\n)+(?=>)/gm, "");
}

// notion-to-md intentionally treats child pages specially and does not emit
// their parent markdown as normal content. For this site, parent pages should
// link to child notes instead of flattening them, so rewrite child_page blocks
// into normal bullet items before serialization.
function collectChildPages(mdBlocks, parentTitle, pages = []) {
  for (const block of mdBlocks) {
    if (block.type === "child_page") {
      const title = getChildPageTitle(block);
      pages.push({
        title,
        slug: toChildWritingSlug(parentTitle, title),
        blocks: block.children ?? [],
      });
      continue;
    }
    if (block.children?.length) {
      collectChildPages(block.children, parentTitle, pages);
    }
  }
  return pages;
}

function linkChildPages(mdBlocks, parentTitle) {
  return mdBlocks.map((block) => {
    if (block.type !== "child_page") {
      return block.children?.length
        ? { ...block, children: linkChildPages(block.children, parentTitle) }
        : block;
    }
    const title = getChildPageTitle(block);
    return {
      ...block,
      type: "bulleted_list_item",
      parent: `- [${title}](/writing/${toChildWritingSlug(parentTitle, title)})`,
      children: [],
    };
  });
}

function renderBody(mdBlocks) {
  const raw = n2m.toMarkdownString(mdBlocks).parent ?? "";
  return fixQuotes(fixCitations(fixTOC(fixToggles(raw))));
}

function buildFrontmatter({ title, date, type, description, tags, listed }) {
  const descYaml = description ? `\ndescription: ${yamlStr(description)}` : "";
  const tagYaml = tags ? `\ntags:\n  - ${tags}` : "\ntags: []";
  const listedYaml = listed === false ? "\nlisted: false" : "";

  return `---
title: ${yamlStr(title)}
date: ${date}
type: ${yamlStr(type)}${descYaml}${tagYaml}${listedYaml}
---

`;
}

// ── Image downloader ──────────────────────────────────────────────────────────
async function downloadImage(url, slug, index) {
  const ext  = url.split("?")[0].match(/\.(\w+)$/)?.[1] ?? "jpg";
  const name = `img-${String(index).padStart(2, "0")}.${ext}`;
  const dir  = path.join(IMG_DIR, slug);
  const dest = path.join(dir, name);

  if (!isDryRun) {
    await mkdir(dir, { recursive: true });
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch image: ${url}`);
    const buf = Buffer.from(await res.arrayBuffer());
    await writeFile(dest, buf);
  }

  return `/assets/images/writing/${slug}/${name}`;
}

// ── Main ──────────────────────────────────────────────────────────────────────
async function fetchAllPages() {
  const results = [];
  let cursor;
  do {
    const res = await notion.search({
      filter: { value: "page", property: "object" },
      sort: { direction: "descending", timestamp: "last_edited_time" },
      page_size: 100,
      ...(cursor ? { start_cursor: cursor } : {}),
    });
    results.push(...res.results);
    cursor = res.has_more ? res.next_cursor : undefined;
  } while (cursor);
  return results;
}

async function syncPage(page, imageCounters) {
  const props       = page.properties ?? {};
  const title       = getProp(props, PROP.title) ?? "(untitled)";
  const slug        = toWritingSlug(title);
  const date        = (page.created_time ?? new Date().toISOString()).slice(0, 10);
  const type        = getProp(props, PROP.type) ?? "note";
  const tags        = getProp(props, PROP.tags);  // single-select -> string or null
  const description = getProp(props, PROP.description);

  // Track per-page image index
  imageCounters[slug] = 0;

  // Override image transformer to download images locally
  n2m.setCustomTransformer("image", async (block) => {
    const raw =
      block.image?.type === "external"
        ? block.image.external.url
        : block.image?.file?.url;
    if (!raw) return "";
    const localPath = await downloadImage(raw, slug, ++imageCounters[slug]);
    const caption = block.image?.caption?.map((c) => c.plain_text).join("") ?? "";
    return `![${caption}](${localPath})`;
  });

  const mdBlocks = await n2m.pageToMarkdown(page.id);
  const childPages = collectChildPages(mdBlocks, title);
  const body = renderBody(linkChildPages(mdBlocks, title));

  if (isDryRun) {
    console.log(`  Would write: ${OUT_DIR}/${slug}.md`);
    console.log(`    title: ${title} | type: ${type} | date: ${date} | tags: ${tags ?? "(none)"} | description: ${description ?? "(none)"}`);
    console.log(`    rendered body: ${body.trim().length} chars`);
    for (const child of childPages) {
      const childBody = renderBody(child.blocks);
      console.log(`  Would write child: ${OUT_DIR}/${child.slug}.md`);
      console.log(`    title: ${child.title} | listed: false | rendered body: ${childBody.trim().length} chars`);
    }
    return;
  }

  await mkdir(OUT_DIR, { recursive: true });
  await writeFile(
    path.join(OUT_DIR, `${slug}.md`),
    buildFrontmatter({ title, date, type, description, tags }) + body,
    "utf8"
  );
  console.log(`  Wrote: ${OUT_DIR}/${slug}.md`);

  for (const child of childPages) {
    const childBody = renderBody(child.blocks);
    await writeFile(
      path.join(OUT_DIR, `${child.slug}.md`),
      buildFrontmatter({
        title: child.title,
        date,
        type,
        tags,
        listed: false,
      }) + childBody,
      "utf8"
    );
    console.log(`  Wrote child: ${OUT_DIR}/${child.slug}.md`);
  }
}

async function main() {
  if (!process.env.NOTION_TOKEN) {
    console.error("Missing NOTION_TOKEN in environment.");
    process.exit(1);
  }

  console.log(isDryRun ? "Dry run -- no files will be written.\n" : "Syncing...\n");

  const pages = await fetchAllPages();
  const published = pages.filter((p) => {
    const status = getProp(p.properties ?? {}, PROP.status);
    return status === PUBLISHED_VALUE;
  });

  console.log(`Found ${pages.length} total pages, ${published.length} marked "${PUBLISHED_VALUE}".\n`);

  if (published.length === 0) {
    console.log(`No pages found with Status == "${PUBLISHED_VALUE}".`);
    console.log("Check that PROP.status names and PUBLISHED_VALUE match your Notion setup.");
    return;
  }

  const imageCounters = {};
  for (const page of published) {
    const title = getProp(page.properties ?? {}, PROP.title) ?? "(untitled)";
    console.log(`Processing: ${title}`);
    await syncPage(page, imageCounters);
  }

  console.log("\nDone.");
}

main().catch((err) => {
  console.error(err.stack ?? err.message ?? String(err));
  process.exit(1);
});
