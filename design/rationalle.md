# SWR > JS fetch > TanStack Query for data fetching

Our end product will be fairly simple, since we are only building a MVP. We **could** use regular JS fetch to fetch data from our API, but since we will be using a lot of the same data in multiple places, it seems like a good opportunity to use a library that will handle the caching for us. SWR is a lightweight library that will handle the caching for us, with a simpler API than TanStack Query. TanStack Query is too complex for our needs, since this is the first time we are using any JS library for data fetching.

# Single-Page App > Multi-Page App

It sounds like single page apps (SPA's) are generally preferred for creating fast and fluid user experiences. The biggest drawback I've read up on is Search Engine Optimization (SEO), but since we are only building a MVP, this is not a concern. I think a SPA for this project will provide a better final experience, and will provide a more applicable learning experience for us for future projects.


