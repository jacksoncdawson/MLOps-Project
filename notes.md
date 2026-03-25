# Project Pitch

**Context:**

The secondhand apparel market is experiencing explosive growth, projected to surpass $230 billion globally in the next few years. However, the independent 'prosumers' and small businesses driving this boom are severely underserved. Legacy trend-forecasting platforms like WGSN and EDITED charge upwards of $20,000 to $30,000 a year, gatekeeping critical data for massive enterprise brands and leaving everyday resellers to rely on intuition and guesswork.

**Our Product:**

We help fashion re-sellers sell and stock inventory faster and in greater quantity by revealing current and upcoming trends in retail. We achieve this by regularly collecting data from various sources, including retail businesses, established historical datasets, and trend-focused platforms like Pinterest. Users can see exactly what is currently trending and what is predicted to spike, based on both historical models and real-time signals. This enables prosumers to strategically target their next haul, anticipate market demand, and streamline the process of sourcing and pricing their inventory with confidence.

## Features:

1. Automatic Listing Schedule

The user specifies the products they currently have in inventory, and our app creates an intelligent listing schedule for each piece, informed by historical, current, and future predicted trends. This enables the seller to optimize the amount they make on their inventory without any time spent researching. For each product, a user-friendly time-series plot is shown, detailing how much they stand to make if they list and sell at the recommended intervals.

> Potential Feature-Add: We could look at similar pieces that sold for more/less, and suggest ways to frame the item on the listing that is most likely to attract buyers.

2. Trend Radar

User is presented with a few time interval options (e.g., 'Today', 'Next Week', '1 Month', '3 Months') which each provide information about trend predictions. Predictions include colors, styles, materials, articles, and vibes.

> Potential Feature-Add: We can connect to retail sites, or listings on Depop/other directly from these prediction pages, to help the user quickly stock anticipatory inventory.

# Data Sources

<div align=center>

| Historical                                                                                                                                             | Regularly Updated                                            | Out-of-Scope   |
| :----------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------- | :------------- |
| [H&M Personalized Fashion Recommendations (Kaggle Dataset)](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview) | 3 - 5 Retail sites (item descriptions + metadata) | Depop/Poshmark |
| [DeepFashion Dataset](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)                                                                          | Pinterest Trends + API with genAI interpretation             | Instagram      |
|                                                                                                                                                        | Reddit                                                       | TikTok         |
|                                                                                                                                                        | Google Trends ('pytrends')                                   |                |

</div>

# 1 Month Timeline

1. Finalize product/feature vision (est. 2 hours)
2. Develop architecture plan (est. 10 hours)
3. Get historical data in GCP (est. 1 hour)
4. Set up automated data collection from available/desired sources (est. 40 hours)
