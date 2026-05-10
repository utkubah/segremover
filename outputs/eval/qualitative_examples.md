# Qualitative examples


## High-confidence REMOVE (p ≥ 0.85)

| genre      |   seg_idx |   p_remove | fn_label          | text_snippet                                                                                                                                 |
|:-----------|----------:|-----------:|:------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|
| commentary |        21 |     0.9359 | discourse_filler  | i just think if the world was ending people wouldn't be like ah well at least i got this shiny rock looks like i'm gonna be a-okay oh no oh  |
| podcasts   |        95 |     0.9363 | off_topic         | So one of which is like this is kind of scary. The idea that um right? You have this you have these uh LMs in broad usage and they're used f |
| tv_series  |        29 |     0.9379 | off_topic         | Yes. But it's hard to do this. It's easier said than done. Yeah.                                                                             |
| podcasts   |        49 |     0.9433 | useful_repetition | and here you know the systems by design are taking uh free free text. Right and this is of course like this is the very biggest strength of  |
| podcasts   |        35 |     0.9711 | discourse_filler  | Or it or discovers like rewrite rule or something. Like these these are keywords. They really trust it with this they sound like keywords. A |
| lectures   |        42 |     0.9713 | off_topic         | Yeah. - So it's a pretty substantial thing.                                                                                                  |
| lectures   |        46 |     0.9735 | off_topic         | Yeah. - But around here, it's pronounced Berlin. - Yeah. -                                                                                   |
| podcasts   |        45 |     0.9744 | useful_repetition | And you can of course adversarially attack that. But there's not really a big harm in it. And you can of course like put some                |
| commentary |         8 |     0.9762 | off_topic         | yeah you want to laugh yeah sure so you know y2k                                                                                             |
| lectures   |        71 |     0.9763 | off_topic         | - Sorry, but they were trying, oh. - Exactly. -                                                                                              |

## High-confidence KEEP (p ≤ 0.15)

| genre      |   seg_idx |   p_remove | fn_label        | text_snippet                                                                                                                                 |
|:-----------|----------:|-----------:|:----------------|:---------------------------------------------------------------------------------------------------------------------------------------------|
| commentary |         1 |     0.0397 | new_information | The newly bowdlerized version of American history played up the contemporary positives of the period, essentially taking Emma Lazarus's sonn |
| lectures   |        61 |     0.0454 | new_information | Now, I talked in my lecture about some of the reasons, the origins of this, the populist movement scared the hell out of white Southern Demo |
| lectures   |        78 |     0.0597 | new_information | - I had neither until I read your book when we hired you. - But the people at the time, when they saw this big bomb go off on Wall Street, t |
| tv_series  |        10 |     0.0645 | new_information | [laughter] And that's THE MOMENT ALSO I realize we make certain noises that only Iranians understand what it means. EVERY TIME HASSAN GOT FO |
| commentary |         0 |     0.0675 | new_information | Don Bluth's 1986 animated film An American Tail tells the story of a young Jewish-Russian immigrant, Fievel Mousekewitz, who is separated fr |
| lectures   |        32 |     0.0718 | new_information | But yeah, her book comes out in 1912. The European World's gonna collapse in this insane all-out war in a matter of only a few months in Eur |
| lectures   |        34 |     0.0731 | new_information | - Sure. - So first of all, the United States was not in the war formally for very long. And we'll talk a little bit more about the actual ex |
| lectures   |        33 |     0.0739 | new_information | It's a moment of the transition of countries, borders, culture, how we even think about mourning, how we think about literature. Film comes  |
| tv_series  |         4 |     0.0751 | new_information | I love this campus. [applause] I'm a Bruin and I tell you, it's so special to be back. Um, I was in college here and we had a little small s |
| lectures   |        59 |     0.0762 | new_information | But even in these surges of peoples coming to the United States from all over the world, to some degree, mainly Europe, but not exclusively, |