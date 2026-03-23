import { defineCollection, z } from "astro:content";

const writing = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    type: z.enum(["note", "post", "reading-note"]),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
    featured: z.boolean().default(false),
    math: z.boolean().default(false)
  })
});

const projects = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    summary: z.string(),
    date: z.coerce.date(),
    status: z.enum(["ongoing", "research", "course-project", "personal"]),
    tags: z.array(z.string()).default([]),
    featured: z.boolean().default(false),
    thumbnail: z.string(),
    role: z.string().default("Researcher"),
    stack: z.array(z.string()).default([]),
    links: z
      .array(
        z.object({
          label: z.string(),
          href: z.string().url()
        })
      )
      .default([])
  })
});

const timeline = defineCollection({
  type: "data",
  schema: z.object({
    year: z.string(),
    title: z.string(),
    body: z.string(),
    category: z.enum(["education", "research", "publication", "project", "next"]),
    accent: z.enum(["ink", "forest", "earth"]).default("ink")
  })
});

const shelf = defineCollection({
  type: "data",
  schema: z.object({
    title: z.string(),
    creator: z.string(),
    type: z.enum(["book", "game", "anime", "film", "paper"]),
    status: z.enum(["now", "finished", "revisit", "favorite"]),
    note: z.string(),
    year: z.string().optional()
  })
});


const publications = defineCollection({
  type: "data",
  schema: z.object({
    title: z.string(),
    authors: z.string(),
    venue: z.string(),
    image: z.string(),
    links: z
      .array(
        z.object({
          label: z.string(),
          href: z.string().url()
        })
      )
      .default([])
  })
});

export const collections = { writing, projects, timeline, shelf, publications };
