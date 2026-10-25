import { getCollection } from 'astro:content'

export async function getSortedProjects() {
  const allProjects = await getCollection('projects', ({ data }) => {
    return import.meta.env.PROD ? data.draft !== true : true
  })
  return allProjects.sort((a, b) => {
    return a.data.published < b.data.published ? 1 : -1
  })
}
