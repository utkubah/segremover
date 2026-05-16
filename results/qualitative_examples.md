# Qualitative examples


## High-confidence REMOVE (p >= 0.85)

| genre      |   seg_idx |   p_remove | fn_label         | text_snippet                                                                                                                                                                                                                 |
|:-----------|----------:|-----------:|:-----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Commentary |        35 |     0.9714 | off_topic        | So you're saying I shouldn't do that? - Yeah, no. Move it carefully.                                                                                                                                                         |
| Commentary |         9 |     0.9719 | off_topic        | and you can still see their eyes and stuff. You know. - There's inter-species relationships happening on my grounds. I won't stand for it.                                                                                   |
| Podcasts   |        60 |     0.975  | off_topic        | no I try it straight I've actually never really drink okay but if you're dentist is upset it's not my fault okay well so Jesse told me to drink this                                                                         |
| Commentary |        55 |     0.9752 | off_topic        | - Why does he talk like that? What is going on? He's shimmying?                                                                                                                                                              |
| Podcasts   |        66 |     0.9754 | discourse_filler | it's the fat between your organs it's the fat that's really bad for you so you can have fat that is sort of on the outside of your body and you can you can grab it you know you can grab it and you can really pinch it spe |
| TED talks  |       116 |     0.9757 | discourse_filler | Hello. It's kind of pioneering and it's also kind of like >> well yeah you know once you hear it                                                                                                                             |
| Podcasts   |        98 |     0.9757 | off_topic        | yeah and nobody teaches you how to deal with this stuff true you have to learn the hard way yeah                                                                                                                             |
| Commentary |        25 |     0.9783 | off_topic        | okay no i'm not okay                                                                                                                                                                                                         |
| Podcasts   |        61 |     0.9787 | off_topic        | straight no she's an expert she said she's a scientist oh gosh yeah you're not supposed to do that Stephen                                                                                                                   |
| Podcasts   |       157 |     0.9804 | clarification    | >> Yeah. >> You know, >>                                                                                                                                                                                                     |

## High-confidence KEEP (p <= 0.15)

| genre         |   seg_idx |   p_remove | fn_label        | text_snippet                                                                                                                                                                                                                 |
|:--------------|----------:|-----------:|:----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Commentary    |        33 |     0.0289 | new_information | But, with chilled rail cars, all this changed, and the country's supply chain reorganized into three parts. First, ranches across the Great Plains funneled their cattle to cities in the Midwest. Taking on the role of ind |
| Commentary    |        24 |     0.0293 | new_information | Well, who's going to be the first to switch back? If any one driver goes back to Route 1 or two, their journey time will be the 25 minutes on the highway plus 20 minutes on the now congested city streets, or 45 minutes i |
| Podcasts      |         1 |     0.0307 | new_information | it's the fat that's really bad for you yeah you're not supposed to do that Stephen oh gosh the glucose goddess is back Jesse enpay is a biochemist and best-selling author with a focus on nutrition and glucose management  |
| Lectures      |         1 |     0.0308 | new_information | Then I'm going to use that to -- I will have a product of matrices and the product that we'll meet will be these elimination matrices and the net result of today's lectures is the big formula for elimination, so the net  |
| Commentary    |        16 |     0.0315 | new_information | On the day, Central Park was turned into a massive festival ground with almost a million people pouring in to see a stacked lineup of performers, including Hallen Oats and the B-52s. But the boldest stunt of the day was  |
| TED talks     |       109 |     0.0337 | new_information | >> Dr. Sammy was amazing and he taught me so much and now I think I want to appreciate beans a little bit more. >> I think I liked Joe Barbosa the best. The guy that was actually running all of the streets of Chicago. >> |
| Podcasts      |        69 |     0.0343 | new_information | Spike so it'll reduce your glucose Spike by up to 30% and your insulin Spike also by up to 30% the way it works is that you have these little scissors in your stomach like miniature scissors called enzymes their job is t |
| Commentary    |        11 |     0.0347 | new_information | A while ago I made a video about a philosopher called Simon Critchley who has this idea called ‘Split Mind Theory’ - the idea is that the human mind is split into two bits: the bit that we experience and then the little  |
| TED talks     |        74 |     0.0347 | new_information | I had to do that a bunch in my 20 plus years of broadcasting, including three years hosting the NPR talk show 1A. Doing that work supercharged my ability to deal with highly problematic people without losing my integrity |
| Entertainment |        42 |     0.0374 | new_information | I know it's not part of it, but. I'll talk, I'll talk to family about it. I have a lot of relatives in Dublin. Is there cheese in it? There, I, I'm just saying that like I know that Indian food has made its way to Irelan |

## False positives — model says REMOVE, gold says KEEP

| genre      |   seg_idx |   p_remove | fn_label             | text_snippet                                                                                                                                                                                                                 |
|:-----------|----------:|-----------:|:---------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Commentary |        47 |     0.7048 | new_information      | A modern version looks something like this. - [Derek] Start by adding a liquid under high pressure to a loop. Passing this liquid through an expansion valve quickly drops the pressure and temperature, causing some of the |
| Podcasts   |        31 |     0.706  | discourse_filler     | for me it helps to break that cycle so if I this weekend I did a re I was I think I was in that cycle of like I wasn't reaching for sugar as in like something sweet I was reaching for like I was having a lot of like toas |
| Commentary |        34 |     0.7076 | redundant_repetition | I hate when this happens. - I think a zipper slider may get stuck if fabric becomes caught in the chain. So if dirt or debris enters in the zipper, the best fix is to carefully remove any trapped fabric or debris, or mov |
| Commentary |         5 |     0.7082 | discourse_filler     | It jammed constantly. And because it was delicate and made from rust prone steel, it actually had to be removed from the garment before you could wash it. So literally unsewn from your skirt. Moreover, if a single hook a |
| Commentary |        36 |     0.7355 | off_topic            | Take a sealed cylinder filled with air and a piston that can move inside it. We'll assume there's no friction or heat lost to the environment. When the piston pushes down, the air is compressed, increasing its pressure a |
| Commentary |        32 |     0.7488 | new_information      | What's an extreme use case for a zipper like this? - Deep sea diving, submarine escape suits. - Submarine escape suits sound really cool. In case of an emergency evacuation of a submarine, you need a suit that can balloo |
| Commentary |        39 |     0.7588 | redundant_repetition | In between the piston and the tank, he placed a big bucket of water and he forced the air through a submerged pipe. This cooled the air down to around the same temperature as the water. The result was that his tank was s |
| Commentary |        36 |     0.7659 | discourse_filler     | - You know what I always wanted to do? Slide down the bowling alley. - I don't know where she got the idea, and maybe from like a cartoon, maybe from that old Disney Channel original movie, Alley Cats Strike, I don't kno |
| Commentary |        25 |     0.7703 | clarification        | But if I grab the pull tab and start pulling, you can see that because of the way that it's shaped, it's actually gonna end up pushing that part up, even though I'm pulling to the side, and that's gonna disengage. You ca |
| Commentary |        33 |     0.7943 | useful_repetition    | But you also need to be able to put it on super quickly. And the best option seems to be this suit with a giant watertight and airtight zipper on the front. Airtight zippers like these even made it onto spacesuits. And t |

## False negatives — model says KEEP, gold says REMOVE

| genre      |   seg_idx |   p_remove | fn_label         | text_snippet                                                                                                                                                                                                                 |
|:-----------|----------:|-----------:|:-----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Commentary |        55 |     0.0383 | discourse_filler | But as you point out, like, it has profound effects. Gorrie was able to make ice because he wasn't afraid to step outside his own profession, medicine, and venture into a new field, thermodynamics. And while you probably |
| Commentary |        30 |     0.0515 | discourse_filler | Oh, and the best email actually suggested we should turn Veritassium's existing content into high performing YouTube videos to increase our audience and extend the brand's reach. That's a genius idea. But unfortunately,  |
| Commentary |         8 |     0.0585 | new_information  | [Gregor] For the next few years, Sundback made minor improvements to Judson's hook and eye design, but none were ever enough to make the product truly functional. Then soon after giving birth to a daughter, his wife Elvi |
| Commentary |        25 |     0.0635 | new_information  | - [Narrator] Any ideas to help you actively find love? - Sex Kung Fu. - [Narrator] I'm sorry, what? - Chinese philosophy with sexual energy meditation. - Do you love him? I don't know what sex Kung Fu is and I never wann |
| Commentary |        27 |     0.0698 | discourse_filler | [Music] okay you're welcome now let's hear a word from today's sponsor this video is sponsored by current folks as technology advances industries usually evolve and adapt over time except banking they refuse to change an |
| Commentary |        21 |     0.0711 | discourse_filler | There we go. I mean, it's a pretty great example of how a more automatic approach can save you a bunch of time, which is what today's sponsor Hostinger is all about. Say you have a bunch of projects spread out over Slack |
| Podcasts   |         2 |     0.0769 | new_information  | yep so what are the hacks the first one is this is a sentence I never thought I'd say in my life um we've just hit 7 million subscribers on YouTube and I want to say a huge thank you to all of you that show up here every |
| Podcasts   |        78 |     0.1059 | new_information  | but I'm hoping that in my lifetime we see some progress because I really do believe that the people in these companies they want to do better they want they don't want people to die they don't want people to get sick but |
| Commentary |        17 |     0.1081 | new_information  | But the city went ahead with it anyway, and no cars were allowed on 42nd Street for the day. Now, to everyone's surprise, the traffic in the surrounding area actually got better. The number of cars was reduced by 20% wit |
| Commentary |         5 |     0.1134 | new_information  | - [Narrator] Welcome to the strangest blind date ever. - Hoping to say goodbye to superficial dating, real life singles sport elaborate makeup and prosthetics to put true blind date chemistry to the test. So I get that t |

## Per-genre failure mode table

One representative FP and FN per genre.

| Genre | Error | p_remove | fn_label | Text snippet |
|---|---|---|---|---|
| Commentary | FP | 0.70 | new_information | A modern version looks something like this. - [Derek] Start by adding a liquid under high pressure t |
| Commentary | FN | 0.04 | discourse_filler | But as you point out, like, it has profound effects. Gorrie was able to make ice because he wasn't a |
| Lectures | FN | 0.16 | new_information | so I take my Matrix bace n now is all 5 5 17 Matrix and now the sub the the question I ask is the su |
| Podcasts | FP | 0.71 | discourse_filler | for me it helps to break that cycle so if I this weekend I did a re I was I think I was in that cycl |
| Podcasts | FN | 0.08 | new_information | yep so what are the hacks the first one is this is a sentence I never thought I'd say in my life um  |

## Transcript-only risk examples

See `transcript_only_risk_examples.md` for visual/deictic segments such as 'this graph', 'on screen', and 'as you can see'.