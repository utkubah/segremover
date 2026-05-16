# Qualitative examples


## High-confidence REMOVE (p >= 0.85)

| genre      |   seg_idx |   p_remove | fn_label         | text_snippet                                                                                                                                                 |
|:-----------|----------:|-----------:|:-----------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Podcasts   |        60 |     0.9847 | off_topic        | oh it's true I realize it I realize that this is serving a purpose which is helping me you know us right                                                     |
| Podcasts   |       132 |     0.9848 | off_topic        | yeah it's really great and it's better than it's ever been um for most of us actually yeah                                                                   |
| Lectures   |         4 |     0.985  | off_topic        | and but also like people seem to like him you know the these suggestions he's making the people saying oh yeah                                               |
| Podcasts   |       106 |     0.985  | discourse_filler | I mean it's like you stole my job yes well yes and I don't think I                                                                                           |
| Podcasts   |       233 |     0.9851 | off_topic        | and I go no I told her no and he literally goes                                                                                                              |
| Podcasts   |        19 |     0.9852 | off_topic        | okay great I understand that or you're a vet okay cool                                                                                                       |
| Commentary |        11 |     0.9864 | discourse_filler | >> No, but like is she Polish? >                                                                                                                             |
| Podcasts   |        58 |     0.9864 | discourse_filler | right they go to the gym yeah there's some they fight in some way yeah boxing UFC whatever                                                                   |
| Commentary |        12 |     0.9868 | discourse_filler | right yeah webs come on it's just webs right totally just webs come on there's no other bodily fluids in this right sorry I got to go come on it's just webs |
| Commentary |         1 |     0.988  | off_topic        | oh no that's hilarious that's so funny I forgot to laugh oh wait                                                                                             |

## High-confidence KEEP (p <= 0.15)

| genre     |   seg_idx |   p_remove | fn_label        | text_snippet                                                                                                                                                                                                                 |
|:----------|----------:|-----------:|:----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Lectures  |       194 |     0.0174 | new_information | With that said, too, well, I'm going to call out CS50's own Brenda Anderson, whose daughter, Sophie, kindly not only created the first incarnation, digitally, of this duck, but also, most recently, once it actually did m |
| Lectures  |        80 |     0.02   | new_information | But in a lot of languages, with accented characters, a lot of Asian characters, this is not nearly enough memory or bits with which to represent all of those possible values. So we need to do a little better than ASCII,  |
| Lectures  |       101 |     0.0209 | new_information | This AI tool that I just saw, I haven't tested it yet, but it seems great in that it gives that-- leads them to the answer without giving it to them the same way that chatGPT says. I always put it like this, though, if y |
| Lectures  |       192 |     0.0211 | new_information | So you need to perturb the output in some way. But within CS50 and within this world of large language models, we do have these tools like ChatGPT, and Bing, chat, and others. And we'll stipulate that, for CS50's purpose |
| Podcasts  |       106 |     0.0215 | new_information | Grand Cormino, wine, spirits. Now, oh wow, I've grabbed this concept of business control ownership and mirrored it with Kevin's drive and entertainment and visibility. Leverage that to get me into the rooms where I may n |
| TED talks |        24 |     0.0218 | new_information | I’m smelling the volatiles that were familiar whereas the machine is smelling -- well, it isn't smelling, it's working out the volatiles from the swab. And we get a result at the end, which has never failed to amaze me.  |
| Lectures  |        69 |     0.0218 | new_information | Especially the control sequences. So now, for these entries, in regions 1 to 2-- it's funny. It has six regions and they're slightly different in their feature. But one thing they have in common is that they are going aw |
| Lectures  |         0 |     0.0219 | new_information | DAVID MALAN: All right. So our first session this morning is going to be with me, with Rongxin, and with Carter. And the goal is to talk really about all things technological to give you a sense of what we, CS50, provide |
| Lectures  |        30 |     0.0219 | new_information | Promise. "The New Orleans Daily Crescent" in the week right after the fall election of 1860, when it was clear Lincoln was gonna be the victor said this in an editorial: "The history of the Abolition or Black Republican  |
| Lectures  |        69 |     0.022  | new_information | So for those who are watching over Zoom, the question was, what are some other popular courses, besides the Python course, for instance, that teachers might be using? We found that both the AI class, led by Brian, and, a |

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