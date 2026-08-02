# Review notes — Paper A

Write one comment per line, tag first. Delete these examples. When done, tell me
"read REVIEW_NOTES.md" (or paste the list in chat).

Tags: ¶N (paragraph), [T#] (table), [F#] (figure), [C#] (contribution), global.

<!-- examples — replace:
¶4 [C3]: make this the lead contribution, it's the most novel
¶18: don't concede "does not beat" so bluntly — say "matches"
[T2]: add greedy pass@10 column
global: cut intro ~30%; too much setup
-->

# Comments
- Title: Should we say when does hyper parameter free decoding breaking during coding? ALso the alpha knobs p^alpha are not generalised form of renyai entropy as I understand. This comment on renayi entropy is applicable to other places in teh document as well.

-  Abstract: Unclear about the meaning of : " We trace both failures to one mechanism—peakedness × hard thresholding". Isn't there a role of reinforcement learning and rewards as well in why this degeneration happens?

- Abstract: This seems unnecessary part of the paper: "and a reproducibility analysis showing how tokenizer/numerics bugs can fabricate decoding-method effects of up to 22pp."

- **¶3** : The original paper does talk about higer moments of alpha, check that, it may not be correct to say: no tunable knobs

- The below parts seem unnecessary: - **[C4]** Code-specific reasoning-loop findings (§5): a paraphrastic-vs-verbatim breakdown on APPS + a verified non-transfer of a published hidden-state precursor to code. **[C5]** A reproducibility post-mortem (§8): three silent bugs shifted pass@1 up to 22pp and fabricated a result that reversed once fixed.
 
 
- **¶13** — But the pass@1 win comes with a diversity deficit — the recurring **pass@1-vs-pass@10 trade-off** --> Does this reflect with CodeBleu and other diversity metrics?

- Can we look at two angles first, there could be two papers may be: The first angle can be: How does pless performing on coding tasks MBPP and HE as compared to other decodring methods: Direct comparison to the papers we have been referring which is : https://arxiv.org/html/2402.06925v3 & https://arxiv.org/html/2507.03160v4, In this we can talk about how pless at different temp(before applying) competes with other decoding methods if need be.

2> The second angle can be, how does pless perform on reasoning models and where alpha in power of p can act as lever (not true renayi entropy)

- Can we get teh crux of the paper and what we want to claim correct first and from there work on other aspects?
