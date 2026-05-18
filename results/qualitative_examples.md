# Qualitative examples


## High-confidence REMOVE (p >= 0.85)

| genre      |   seg_idx |   p_remove | fn_label         | text_snippet                                                                                                              |
|:-----------|----------:|-----------:|:-----------------|:--------------------------------------------------------------------------------------------------------------------------|
| Podcasts   |       106 |     0.9744 | discourse_filler | I mean it's like you stole my job yes well yes and I don't think I                                                        |
| Commentary |        35 |     0.9745 | off_topic        | (laughs) There could have been another name for it. Or there could have just not been any name at all. That's a hot take. |
| Podcasts   |        19 |     0.9747 | off_topic        | Hello. Thank you. You can come on in. Can I have a go, please? Yes.                                                       |
| Commentary |        30 |     0.9768 | off_topic        | okay [Music] what the [ __ ] whatever all right okay                                                                      |
| Commentary |        18 |     0.9782 | off_topic        | oh man all right well i'm out of here bye love you okay bye i love you                                                    |
| Podcasts   |        81 |     0.9785 | off_topic        | and you can only get this online for limited time so make sure you don't miss out [Music] [Music]                         |
| Podcasts   |       190 |     0.9787 | discourse_filler | yeah it's it's it's hard to fathom actually yeah                                                                          |
| Commentary |         1 |     0.9789 | off_topic        | oh no that's hilarious that's so funny I forgot to laugh oh wait                                                          |
| Podcasts   |       162 |     0.9805 | off_topic        | That's obviously kind of reasonable. Yeah. I yeah I wouldn't read too much into that.                                     |
| Commentary |        11 |     0.981  | off_topic        | anymore I don't want to date him or talk to him okay yeah                                                                 |

## High-confidence KEEP (p <= 0.15)

| genre         |   seg_idx |   p_remove | fn_label        | text_snippet                                                                                                                                                                                                                 |
|:--------------|----------:|-----------:|:----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Entertainment |        14 |     0.0202 | new_information | Speaker 2: So I understand that in college you do weekly hustle days, which was bringing your music on the train with you and then sitting outside record executives doors for hours on end. Yeah. What stands out is like y |
| Entertainment |       182 |     0.0207 | new_information | The plan that it was trying to achieve is to bring back sovereignty to the indigenous population. When you dig in this line, you find Jewish history. This is where we come from. Now there are individuals that use the ter |
| Podcasts      |       194 |     0.021  | new_information | but like let's throw some stuff up higher, bigger battery, get rid of the SIM card tray, like I guess getting rid of a SIM card tray isn't taking advantage of everything. But >> they did for the first time in Apple histo |
| Podcasts      |        16 |     0.021  | new_information | You can share like analytics with them and they can see like how the video is doing and things like that. So, I've had a couple older videos like the collab we did with the Dyson headphones with Dr. Mike where he comes i |
| Entertainment |       137 |     0.0213 | new_information | Trump is forcing us to decide whether or not we should feed our country or whether or not we should provide our country with basic access to health care. >> So, and the idea is that he has influence over Republicans. So, |
| Entertainment |         8 |     0.0215 | new_information | No way. His birthday isn't until tomorrow, but we felt like he deserves some appreciation today since it's Friday. Well, how about we celebrate my big day with some terrible culinary offenses. It's time for Food Crimes:  |
| Lectures      |       120 |     0.0218 | new_information | So we covered a lot of ground already. So we talked about zero sum games. We talked about pure Nash equilibrium and how sometimes they don't exist. We extended our solution concept to mixed Nash equilibrium. And once we  |
| Entertainment |         1 |     0.0221 | new_information | And so does Cook Unity. Their meals arrive at your door in sustainable packaging and are made with responsibly sourced ingredients and they're ready to enjoy in minutes. Head to the link in our description now to try Coo |
| Podcasts      |        98 |     0.0224 | new_information | > I think when they say that you can also infer that the CPU improvements are not as big and not as much of a focus >> and that the stock price will go up. But yeah that they for people who have ondevice AI processing wo |
| Podcasts      |         1 |     0.0225 | new_information | And there's one key thing that's given much better returns than any real estate, than any stock, and even any cryptocurrencies. So, let's talk about the real way to build true wealth. Jaspreet Singh is the no-nonsense fi |

## False positives — model says REMOVE, gold says KEEP

| genre      |   seg_idx |   p_remove | fn_label             | text_snippet                                                                                                                                                                                                                 |
|:-----------|----------:|-----------:|:---------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Commentary |        64 |     0.7024 | new_information      | - All of it in one device. - So it's not actually lithium that burns. A modern lithium-ion battery like this one here contains very little lithium, ironically. It's actually everything else inside here that's dangerous.  |
| Commentary |         5 |     0.7029 | discourse_filler     | It jammed constantly. And because it was delicate and made from rust prone steel, it actually had to be removed from the garment before you could wash it. So literally unsewn from your skirt. Moreover, if a single hook a |
| Commentary |        41 |     0.7036 | redundant_repetition | And here at this point, they intersect. Now since electrons behave as waves, the waves from the intersecting beams overlap and produce an interference pattern, bright fringes with gaps in between them. The exact pattern  |
| Commentary |        53 |     0.7036 | useful_repetition    | Yeah. Other snapping structures could also switch upon resonance. But the problem that once they switch, they're much more elongated, much more contracted. So they don't, let's say, they wouldn't provide the same functio |
| Commentary |        10 |     0.7043 | clarification        | Now, this is especially true on a real size zipper where it's practically impossible. But if I add this slider to the bottom and try pulling on the pull tab here, suddenly it's effortless. So how does it do it? Well, I c |
| Commentary |        13 |     0.7046 | clarification        | So why is this? - At room temperature, rubber's polymer chains are constantly vibrating and bumping into each other. And other smaller molecules like air molecules or trapped water molecules also jostle around and bump i |
| Commentary |        18 |     0.7053 | new_information      | The chain itself bounces back. This is where rubber gets its elasticity from. So natural rubbers straight from the tree is already stretchy. It's also waterproof since all those chains are just a bunch of carbon and hydr |
| Commentary |        41 |     0.7075 | new_information      | And they began to wonder. What if you could take a second off of every transaction? What about two? What if you could make each transaction basically instant without having to make contact at all?                         |
| Commentary |        86 |     0.7088 | redundant_repetition | The material can't go back to the same shape after. And if you keep going, it will fracture. But the stress-strain curve for rubber is a little weird. It starts out stretching very far with very little force. That's the  |
| Commentary |         5 |     0.7091 | discourse_filler     | I mean, what would that even mean? So, in classical physics, the solution is simple. You just ignore the negative energy solutions. They can't physically represent anything, right, Casper? -                               |

## False negatives — model says KEEP, gold says REMOVE

| genre      |   seg_idx |   p_remove | fn_label         | text_snippet                                                                                                                                                                                                                 |
|:-----------|----------:|-----------:|:-----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Commentary |        67 |     0.0299 | new_information  | For example, Iran controls a single strait that handles roughly 20% of the world's oil. What happens when they threaten to close it amid US strikes on their nuclear sites? Well, on the topic, USA today ran oil hits five- |
| Commentary |        69 |     0.0379 | new_information  | Studio I had another Prime hydration drink and the most incredible thing happened I think my body was finally getting used to this diet because I had a reunion with my long lost friend the toilet and I won it only took 3 |
| Commentary |        21 |     0.038  | new_information  | That's not a healthy way to live your life. That's why I opened Buddy's Tongue Rubbers. Finally, you can have an elegant meal while a trained professional strokes that pink mel form of yours after every succulent bite. O |
| Commentary |        62 |     0.0399 | new_information  | They compile news from outlets all over the world into one place so that you can easily see the partisan split and with their color-coded layout, it's also easy to sort your news by factuality, ownership and source so th |
| Commentary |        55 |     0.0438 | discourse_filler | But as you point out, like, it has profound effects. Gorrie was able to make ice because he wasn't afraid to step outside his own profession, medicine, and venture into a new field, thermodynamics. And while you probably |
| Commentary |        98 |     0.0444 | new_information  | and then they got to keep coming back for more right therefore increasing the amount of sales you get therefore increasing the amount of money in your pocket in your hot pocket I can't do that okay because I only sell me |
| Lectures   |         1 |     0.0459 | new_information  | And a little about me. I did my PhD here at Stanford from 2012 to 2018, working with [INAUDIBLE] on deep learning, computer vision, pretty much all tasks in computer vision around that time. During my time here at Stanfo |
| Commentary |        25 |     0.0463 | new_information  | - [Narrator] Any ideas to help you actively find love? - Sex Kung Fu. - [Narrator] I'm sorry, what? - Chinese philosophy with sexual energy meditation. - Do you love him? I don't know what sex Kung Fu is and I never wann |
| Commentary |        45 |     0.0474 | new_information  | The magnetic field could be just zero, and yet the presence of some vector potential could actually lead to observable effects. That wasn't supposed to happen, right? - What I love about this story is that it reminds me  |
| Commentary |        30 |     0.0474 | discourse_filler | Oh, and the best email actually suggested we should turn Veritassium's existing content into high performing YouTube videos to increase our audience and extend the brand's reach. That's a genius idea. But unfortunately,  |

## Per-genre failure mode table

One representative FP and FN per genre.

| Genre | Error | p_remove | fn_label | Text snippet |
|---|---|---|---|---|
| Commentary | FP | 0.70 | new_information | - All of it in one device. - So it's not actually lithium that burns. A modern lithium-ion battery l |
| Commentary | FN | 0.03 | new_information | For example, Iran controls a single strait that handles roughly 20% of the world's oil. What happens |
| Entertainment | FP | 0.73 | discourse_filler | I just need broth and something very. It cleans you out. Settling. It cleans you out. For my sour tu |
| Entertainment | FN | 0.05 | new_information | What are you talking about? We decided we were going to make this before you brought like 80 bucks w |
| Lectures | FP | 0.72 | redundant_repetition | And in these little local neighborhoods of our input images. So we can play this trick on the first  |
| Lectures | FN | 0.05 | new_information | And a little about me. I did my PhD here at Stanford from 2012 to 2018, working with [INAUDIBLE] on  |
| Podcasts | FP | 0.73 | new_information | I have these phases especially if I'm really busy and I can't do all the hacks all the time where I' |
| Podcasts | FN | 0.05 | new_information | yep so what are the hacks the first one is this is a sentence I never thought I'd say in my life um  |
| TED talks | FP | 0.71 | new_information | and I was like somebody who listen to me so I said so I have this idea and I wrote one long windy em |
| TED talks | FN | 0.21 | off_topic | yeah but we are nicer like I have lots of oil yeah but I've got lions like it just every Co and I kn |

## Transcript-only risk examples

See `transcript_only_risk_examples.md` for visual/deictic segments such as 'this graph', 'on screen', and 'as you can see'.