# Product

## Register

product

## Users

Simon and the developers working on passphoto-validation. They open a check report from the API notebook or with curl, on a desk screen next to their editor and terminal, often for long testing stretches. Their job: decide whether the pipeline judged a photo right, which means seeing every rule, every check, every measurement and the reason behind each status.

## Product Purpose

passphoto-validation checks a photo against the German ID photo rules (the Fotomustertafel) and says, rule by rule, whether it passes and why. The HTML report is the human view of one `/v1/check` answer. It succeeds when a developer sees the verdict, what failed and the number that made it fail within seconds, and can then drill into any rule without leaving the page.

## Brand Personality

Brutalist, analytical, exact. It reads like an analyst's console: dense, gridded, data first, nothing ornamental. Trust comes from exposing the raw data, not from friendliness.

## Anti-references

- SaaS cards: rounded corners, soft shadows, tinted pill badges.
- Side-stripe callouts (a coloured left border on a box).
- Decorative icons, gradients, hero metrics, friendly consumer copy.
- Anything that hides data the JSON has, or renames it.

## Design Principles

1. Data is the interface: show the measurements and reasons as the API returns them, with the same names.
2. Problems first, evidence second: verdict, then what failed and why, then everything else.
3. Structure from the grid and the type, not from boxes and colour.
4. Status reads without colour: every status has its own code and texture.
5. One page, one file: it must work saved, offline and inside a notebook frame.

## Accessibility & Inclusion

WCAG 2.2 AA. Status is never told by colour alone (codes and textures). Keyboard reachable: every rule opens with the keyboard, and the overview links to each rule. No motion.
