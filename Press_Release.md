# Congressional Races Are Decided by Demographics, Not Polls; A New Dataset Is Built Around That Fact

## Control of Congress Was Decided by 7,000 Votes. Nobody Saw It Coming.

In 2024, control of the US House of Representatives came down to three congressional districts. The winning margins in those races totaled roughly 7,000 votes out of nearly 148 million cast nationwide. Most forecasters, relying on the same polls they always use, did not see the outcome coming.

That is not bad luck. It is a predictable consequence of looking in the wrong place. Polls tell you what voters say they will do a few weeks before an election. Demographics tell you why a district has been moving for years. This project is built on the belief that the second kind of information is more powerful, and that nobody has put it together properly until now.

## The Problem with How We Predict Elections

Every two years, campaigns spend millions of dollars, journalists write thousands of articles, and forecasters build elaborate models, all centered on one thing: polls. The problem is that polls at the congressional district level are expensive to conduct, infrequent, and historically unreliable in the races that actually matter.

Of the 435 House races in 2024, the average margin of victory was over 27 percentage points. Most districts are not competitive at all. The roughly 50 to 60 that genuinely are, the ones that decide which party runs the chamber, are the ones where polls tend to be the most wrong.

Meanwhile, the information that actually explains why a district is moving sits largely unused in public databases. When a neighborhood becomes more college educated, when its median age drops, when its racial composition shifts, the political consequences tend to follow within one or two election cycles. These are not surprises. They are trends that have been building in plain sight.

No one has built a dataset that systematically combines these demographic trends with the full history of election results and the broader national environment, until now.

## A Smarter Way to See Which Races Are Actually in Play

This project assembles a longitudinal dataset covering all 435 US House districts from 2012 through 2022. It draws on three sources of public information: official certified election results from MIT, demographic data from the US Census Bureau, and national economic and political indicators including unemployment, economic growth, and presidential approval ratings.

The result is a dataset that can answer questions that polling cannot. Which districts have been quietly shifting toward one party over the past decade? Which ones are structurally competitive regardless of who is on the ballot? When the national mood turns against the party in power, which seats are the first to flip?

A predictive model built on this dataset identifies which districts will change party hands with a high degree of accuracy, and it does so without using a single poll. The strongest signals are ones that are available months or even years before election day: how wide the winning margin was in the prior election, which party the national environment is favoring, and how voters have been trending in the district over time.

## Where the Competitive Races Are, and Why It Matters

The chart below shows how many House seats each party won in each election year from 2012 through 2022, alongside the national political environment measured by the generic congressional ballot. The pattern is clear. In years when one party had a strong national advantage, seats flipped. In more neutral years, fewer did.

What the chart does not show, but what this dataset makes visible, is which specific districts were in play and why. A district that has become significantly more college educated over the past decade is more likely to be competitive than it was before, regardless of how it voted in the last cycle. A district where the winning margin has narrowed in three straight elections is a different kind of target than one where the margin has held steady.

The dataset makes this kind of structural analysis available to anyone: campaigns deciding where to compete, journalists trying to explain why a race is close, researchers studying how American communities are changing and what those changes mean for representation.

<img width="1480" height="730" alt="fig1_seats_by_year" src="https://github.com/user-attachments/assets/f6e75974-833f-466b-a6f4-a1812c819d61" />

*(Chart: fig1_seats_by_year.png -- House seats won by party per election year with national generic ballot overlay, generated from this dataset)*

Understanding which districts are genuinely in motion is not just useful for winning elections. It is useful for understanding American democracy. When competitive races are identified early, they attract more candidates, more coverage, and more voter attention. The goal of this project is to make that identification more accurate, more transparent, and more accessible than it has ever been.
