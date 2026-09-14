from __future__ import annotations

import base64
import json
import mimetypes
import re
import subprocess
import time
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path

from PIL import Image


SYSTEM_PROMPTS = {
    "rave": """Look at the image and write one concise prompt for two seconds
of clearly visible scene motion and lighting or atmospheric effects. Describe
contained dance-like motion while the subject keeps the same identity, location,
and overall composition. Prefer in-place body rhythm: a hip sway, shoulder roll,
chest bounce, torso dip, pose pulse, or dramatic battle stance. Pair the body
motion with visible hair and clothing fabric movement when applicable. Never
describe walking, running, jumping, stepping, turning away, leaving the pose, or
single-limb-only motion like moving one arm. Never mention camera movement,
panning, reframing, zooming, or focus changes.

Use one clearly perceptible dance/body motion plus hair or loose fabric motion.
Add one or two effects that visibly evolve through the shot: flashing neon,
shifting colored light, curling fog, smoke, dust, sparks, rain, or pulsing
energy. Preserve every subject, object, outfit, and background. No scene change
and no new objects. Avoid words such as subtle, barely, faint, tiny, still,
static, or imperceptible. Use 16-30 words. Return only the prompt.

Good examples:
The woman sways her hips in place as long hair and loose fabric whip gently
while neon signs flare behind her.

The green fighter pulses into a battle stance as jacket fabric snaps, hair
shakes, and glowing dust swirls through the air.

The armored woman rolls her shoulders in place as cape edges ripple, hair
flicks, and energy sparks flash near her feet.

The pink-haired woman bounces rhythmically in place as loose hair tips and skirt
fabric sway while green energy bands pulse.""",
    "little-queen": """Write one 20-40 word image-to-video prompt for two
seconds of unmistakably high-energy magical-girl or anime-boss motion.

First silently inspect the subject count, crop, starting pose, body parts present,
held props, clothing, and background. Never invent a physical crown, weapon,
garment, building, creature, or scene detail. Temporary luminous overlays are
allowed and encouraged: energy armor, glowing gauntlets, aura wings, halos,
spell circles, beams, flames, shockwaves, and transformation light are effects,
not new physical objects.

The required creative mode must remain obvious even when choreography is
adapted to the image. Begin from the exact pose and remain in place. Use larger
torso, shoulder, pose, or prop motion for three-quarter and full-body images;
use a crown or prop only when present; otherwise use the chest, expression,
one clearly present hand, or silhouette for close crops.
Both hands may act together only when both are clearly shown. Continue seated
or airborne poses instead of standing or planting unseen feet.

Describe one coherent two-stage arc: energy charges, traces, or compresses,
then it releases, erupts, locks into a transformed form, or unfolds at a clear
peak. The two stages must be one evolving effect system, not unrelated effects.
Hair or fabric reaction is optional and should appear only when it strengthens
the main action; never use it as template filler.

Keep identity, physical objects, location, and composition. No walking,
running, leaping, lunging, stepping, spinning, turning away, camera motion,
reframing, zoom, focus change, or scene change. Never settle for a head tilt,
smile, quiet pose, gentle glow, simple prop brightening, or effect that merely
expands. Avoid subtle, barely, faint, tiny, still, static, and imperceptible.

Good examples, only when their physical anchors are shown:
Luminous armor lines trace across her dress, then lock into radiant battle
plates as one rainbow shock ring bursts from her silhouette.

She compresses a white-hot aura around her current pose, then erupts upward in
a roaring energy column as one circular shockwave blasts outward.

She charges a star core before her silhouette, then fires one heart-bright beam as
the surrounding aura snaps backward.

Return exactly two plain-text lines with no markdown or commentary:
TWO_HANDS_CLEAR: yes or no
PROMPT: the 20-40 word prompt

Set TWO_HANDS_CLEAR to yes only when two distinct hands are clearly present in
the source image. The PROMPT may say "both hands" only when that answer is yes.""",
}

SYSTEM_PROMPT = SYSTEM_PROMPTS["rave"]

FALLBACK_PROMPTS = {
    "rave": (
        "The subject blinks softly as loose hair or fabric stirs, existing "
        "lights pulse, and faint atmospheric particles drift through the scene."
    ),
    "little-queen": (
        "Luminous battle-armor lines trace across the little queen's silhouette, "
        "then lock into radiant plates as one rainbow shock ring bursts outward."
    ),
}

FALLBACK_PROMPT = FALLBACK_PROMPTS["rave"]

LITTLE_QUEEN_MODE_SCHEDULE = (
    "transformation",
    "power-up",
    "attack",
    "transformation",
    "power-up",
    "environmental-spell",
    "transformation",
    "attack",
    "power-up",
    "transformation",
    "attack",
    "power-up",
    "transformation",
    "environmental-spell",
    "attack",
    "power-up",
    "transformation",
    "attack",
    "power-up",
    "transformation",
)

LITTLE_QUEEN_MODE_DIRECTIVES = {
    "transformation": (
        "Create an obvious magical transformation. Energy must trace or scan "
        "across her, then visibly lock, unfold, or resolve into a luminous "
        "armor, wing, mantle, or final-form overlay."
    ),
    "power-up": (
        "Create a Dragon Ball Z-scale power-up. Energy must compress or gather "
        "around her current pose, then erupt as a roaring aura column, overload, "
        "or forceful shockwave. A quiet glow is not enough."
    ),
    "attack": (
        "Create one signature attack. Energy must charge at a present prop, "
        "crown, chest focus, or clearly present hand configuration, then "
        "fire, launch, slash, or release with a forceful payoff."
    ),
    "environmental-spell": (
        "Create an aggressive environmental spell using one present, neutrally "
        "named scene anchor. If no anchor is certain, use existing background "
        "light or air. Energy must race through that anchor, "
        "converge on her, then burst into a controlled spell wave or sigil."
    ),
}

LITTLE_QUEEN_MODE_FAMILIES = {
    "transformation": (
        "battle-armor ignition: luminous contour lines scan across her outfit, then armor-light plates snap into a final form",
        "rainbow fire wing awakening: compressed backlight splits open into enormous energy wings and a hot halo",
        "phoenix mantle reveal: feather-shaped fire traces her silhouette, then locks into a blazing aura mantle",
        "prism exosuit boot-up: an energy wireframe races over her pose, then resolves into bright gauntlet and shoulder overlays",
        "butterfly battle metamorphosis: pastel light plates unfold from her silhouette into a sharp winged armor form",
        "royal coronation overload: crown light scans downward, then assembles a radiant royal battle-form overlay",
        "aurora final-form reveal: aurora bands wrap her silhouette, then snap open as an upgraded luminous costume aura",
        "shadow-to-rainbow cleanse: dark energy peels away, then converts into fierce rainbow armor light",
    ),
    "power-up": (
        "boss-fight aura overload: tight energy rings compress around her, then erupt vertically with a circular shockwave",
        "dragon-prism aura roar: light coils around her current pose, then surges upward as a roaring dragon-shaped aura",
        "supernova core charge: a white-hot core condenses at her center, then detonates into a controlled starburst aura",
        "elemental rainbow eruption: colored energy gathers below her silhouette, then blasts upward as a towering flame column",
        "gem-heart reactor overload: a present jewel or chest focus pulses faster, then releases a full-body aura surge",
        "solar crown power-up: crown light compresses into a tiny sun, then sends a violent aura wave down her silhouette",
        "floor-sigil ascension: luminous geometry channels energy upward, then erupts around her as a final power state",
        "storm mantle overdrive: charged light gathers around her outline, then breaks into lightning and a pressure-ring blast",
    ),
    "attack": (
        "beam-cannon release: energy compresses at a present focus, then fires as one thick rainbow beam with recoil light",
        "heart-star finisher: a heart-star core charges, then launches as a brilliant projectile with one impact ring",
        "moon-prism shield burst: a barrier condenses before her pose, then slams outward as a forceful counterwave",
        "crescent slash release: energy traces a present prop or crown focus, then launches as one cutting moon arc",
        "rainbow drill cannon: spiraling light tightens into a hot core, then releases as one piercing corkscrew beam",
        "phoenix beam finisher: feather-light gathers into a bird-shaped core, then surges outward as one blazing attack",
        "royal shockwave strike: energy compresses close to her silhouette, then bursts outward as one crown-shaped pressure wave",
        "constellation lance: star-map light traces from a present focus, then locks into and launches one piercing energy lance",
    ),
    "environmental-spell": (
        "royal sigil chain: light races through one present structure or plant, converges around her, then bursts as a large sigil",
        "garden spell eruption: energy traces present flowers, vines, or branches, then detonates into one petal-shaped spell wave",
        "crystal resonance spell: present gems or bright structures pulse in sequence, then discharge one faceted energy surge",
        "celestial scene command: present stars, ornaments, or lights align around her, then release one focused royal wave",
        "portal-rune invocation: runes trace across a present background plane, converge behind her, then open into one energy blast",
        "elemental scene reversal: light drains from one present scene anchor, spirals toward her, then erupts in transformed color",
    ),
}

LITTLE_QUEEN_MODE_FALLBACKS = {
    "transformation": (
        "Luminous battle-armor lines trace across the little queen's silhouette, "
        "then lock into radiant plates as one rainbow shock ring bursts outward."
    ),
    "power-up": (
        "The little queen braces in place as a roaring rainbow aura compresses "
        "around her silhouette, then erupts upward and releases one circular shockwave."
    ),
    "attack": (
        "The little queen charges a star-bright core before her silhouette, then "
        "releases one focused rainbow beam as the surrounding aura snaps backward."
    ),
    "environmental-spell": (
        "Existing background light gathers into a luminous royal sigil around "
        "the little queen, then the spell erupts across the scene as one controlled wave."
    ),
}

LITTLE_QUEEN_MOVE_FAMILIES = (
    "ultimate armor transformation: glitter armor plates lock onto her dress, gauntlets, shoulders, or crown",
    "rainbow fire wing summoning: wings flare open behind her while hair and cape surge outward",
    "beam cannon charge-up: both hands cup a growing orb before releasing or aiming a rainbow beam",
    "moon-prism shield burst: she braces low as a luminous shield or barrier expands around her",
    "scepter or weapon invocation: she raises a visible wand, sword, staff, pistol, or crown focus into a spell circle",
    "crystal power-up: crystals, gems, towers, or floor symbols feed energy into her full body",
    "heart-star attack: heart and star energy gathers at her chest, palms, crown, or weapon before firing",
    "boss-fight aura transformation: aura rings, halos, and armor light stack around her in a dramatic power-up",
    "elemental rainbow fire eruption: fire, sparks, and glowing wind surge from the floor around her planted body",
    "two-hand casting choreography: both hands carve a large spell shape while sleeves, hair, and cape snap outward",
    "royal coronation overload: her crown blooms into a radiant halo while shoulder armor and skirt armor assemble",
    "gauntlet ignition: glowing gauntlets form over both hands as she compresses energy between her palms",
    "starfall command pose: she points a visible weapon or crown focus skyward as falling star sparks answer",
    "phoenix mantle transformation: a fiery cape or wing mantle unfurls behind her as her silhouette brightens",
    "mirror shield counterspell: she braces behind a reflective magic pane that catches and redirects light",
    "ribbon seal release: magical ribbons whip into sigils around her while she opens a sealed power form",
    "orb compression charge: she squeezes a swirling orb smaller and brighter between both hands",
    "meteor hammer spell: a chain of glowing star-orbs whirls around her planted body before striking outward",
    "butterfly armor metamorphosis: luminous wing plates unfold from her back and shoulders like a battle costume",
    "crown beam alignment: her crown, palms, and a background tower align into one focused beam path",
    "rose-thorn barrier bloom: thorny rose light spirals up from the floor into a protective royal barrier",
    "dragon-prism aura roar: a dragon-shaped aura coils behind her while she locks into a fearless combat pose",
    "constellation blade awakening: a sword, wand, or staff becomes a star map as light traces across the weapon",
    "halo ring launcher: floating rings stack before her palms and fire forward like a magical cannon",
    "floor sigil ascension: a spell circle lifts light around her boots, dress, cape, and crown without moving her location",
    "candy comet barrage: she directs small comet-like sweets or starlets from both hands toward the background",
    "gem-heart reactor charge: a gem on her outfit, crown, or prop pulses into a bright heart-shaped reactor",
    "rainbow drill beam: spiraling color bands twist from her hands or weapon into a focused piercing beam",
    "aurora veil transformation: aurora sheets wrap around her body and snap open as upgraded armor light",
    "throne-room command burst: she gestures regally with a visible prop as royal sigils explode around her",
    "spellbook page cyclone: glowing pages or rectangular light panels orbit her like a magic manual opening",
    "crescent moon finisher: a crescent-shaped arc forms behind her before launching from her palms or weapon",
    "solar crown flare: the crown flashes like a tiny sun and sends rings of armor light down her body",
    "bubble prism charge: floating bubbles fuse into a dense prism orb held between both hands",
    "storm-cape overdrive: her cape becomes a snapping storm banner as lightning and glitter surge along its edge",
    "royal mecha armor snap-on: toy-like magical armor segments pop into place over shoulders, boots, and gloves",
    "garden spell bloom: flowers, vines, or petals burst into a magic circle that fuels her attack pose",
    "ice crystal queen mode: crystalline frost armor and rainbow refractions grow outward from her crown or weapon",
    "lava-candy forge-up: molten candy-colored fire forges glowing gauntlets or boots around her planted body",
    "clockwork star transformation: rotating clock-star rings lock into position around her as armor light powers up",
    "portal beam summoning: a circular portal opens behind her and feeds energy into both hands",
    "tiny familiar power link: a visible creature, plush, or mascot-like element sends energy into her attack",
    "waterfall prism surge: liquid rainbow streams rise around her and form wing or armor shapes",
    "shadow-to-rainbow cleanse: dark aura peels away from her silhouette and transforms into rainbow armor light",
    "firework cannon pose: she aims both hands or a prop as layered fireworks charge into one controlled blast",
    "gem tower resonance: towers, gems, or skyline lights pulse in sequence into her crown and palms",
    "supernova curtsy power-up: a royal curtsy-like dip becomes a compact starburst without stepping away",
    "magic bow draw: she pulls a glowing bowstring of light between both hands and aims a rainbow arrow",
    "crown constellation summon: crown jewels project a constellation animal or crest behind her body",
    "royal drumbeat shockwave: visible floor or background lights pulse in rings as she pounds energy downward with both hands",
    "plasma tiara throw: a tiara-shaped arc spins from her crown focus while she holds a dramatic launch pose",
    "rainbow exosuit boot-up: an energy exosuit wireframe scans over her body and resolves into glitter armor",
    "spell lance formation: a lance of light forms from her palms or weapon while cloak, hair, and ribbons stream backward",
    "moon rocket charge: crescent sparks spiral upward like a launch plume while she charges an upward beam",
    "royal final-form reveal: layered halos, wings, armor, and weapon glow reveal a brief ultimate form silhouette",
    "prism cage breaker: she breaks open a cage of light from inside, sending shards outward while staying centered",
    "candy storm monarch mode: candy sparks and star confetti orbit her as a queenly aura intensifies",
    "nebula heart reactor: a tiny galaxy-heart spins in front of her chest before expanding into an attack aura",
    "rainbow phoenix beam: a phoenix-shaped beam forms from her hands or wand and surges toward the scene detail",
)

LITTLE_QUEEN_EFFECT_FAMILIES = (
    "rainbow fire wings and hot glitter embers",
    "glitter armor plates and star-shaped gauntlet light",
    "heart-star beam and pink-gold shockwaves",
    "moon-prism aura and silver-blue spell rings",
    "crystal sparks and faceted rainbow refractions",
    "effervescent rainbow orb and bubble-like light motes",
    "candy-colored flame columns and crown halo flares",
    "prismatic lightning used sparingly with a secondary armor or wing effect",
    "celestial ribbon trails and rotating magic sigils",
    "beam cannon glow and rippling circular shock rings",
    "gold crown halo, shoulder armor flashes, and pink comet sparks",
    "aurora sheets, jewel refractions, and soft explosive shimmer",
    "rose-gold thorn rings, petal sparks, and royal barrier light",
    "phoenix-feather flames, orange-pink wing heat, and glitter ash",
    "mirror-bright shield facets, reflected beams, and silver shard sparkles",
    "candy comet trails, gumdrop-colored sparks, and tiny star impacts",
    "clock-star rings, ticking crescent lights, and blue-white gear halos",
    "nebula heart glow, galaxy speckles, and purple-pink aura bands",
    "crystal tower resonance, faceted beams, and gem-spark rain",
    "spellbook page panels, gold script glyphs, and luminous paper trails",
    "butterfly wing plates, pastel armor shimmer, and pollen-like glitter",
    "dragon-prism aura coils, scale-shaped sparks, and roaring light arcs",
    "crescent moon slash, silver-blue trails, and midnight sparkle dust",
    "bubble prism orb, effervescent motes, and rainbow lens flares",
    "lava-candy fire, molten pink highlights, and forged gauntlet sparks",
    "ice crystal armor, frost halos, and rainbow refraction shards",
    "portal backlight, circular runes, and energy streams feeding her hands",
    "waterfall prism ribbons, liquid light wings, and splashing star motes",
    "shadow peeling into rainbow flame, clean white highlights, and aura sparks",
    "firework bloom layers, controlled blast rings, and glitter smoke curls",
    "constellation animal silhouette, crown-jewel beams, and star map lines",
    "magic bowstring flare, rainbow arrow trail, and expanding impact halo",
    "tiara-shaped plasma arc, gold-pink spin trail, and crown flare",
    "wireframe exosuit scan, neon armor seams, and glitter boot-up pulses",
    "spell lance glow, cloak-edge streaks, and piercing rainbow rays",
    "moon rocket plume, upward crescent sparks, and vertical aura flames",
    "ultimate-form silhouette, stacked halos, and final armor glints",
    "light cage shards, prism fragments, and expanding freedom shockwave",
    "candy storm confetti, monarch aura, and star-sugar explosions",
    "royal drumbeat rings, floor shockwaves, and pulsing bass-light glyphs",
    "gem-heart reactor pulse, chest halo, and synchronized tower flashes",
    "solar crown flare, sunburst armor rings, and warm lens sparks",
    "meteor hammer star-orbs, orbit trails, and bright impact glitter",
    "rainbow drill spiral, corkscrew beam bands, and hot white core light",
    "royal sigil mandala, pink-gold glyphs, and expanding crown-shaped rays",
    "familiar-link sparkles, mascot energy beam, and protective heart motes",
    "garden bloom magic, flower-petal sigils, and emerald-violet leaf sparks",
    "storm-cape lightning, glitter rain, and high-contrast silhouette flashes",
    "prism phoenix outline, feather-shaped beam tongues, and radiant tail sparks",
)

LITTLE_QUEEN_V7_EFFECT_FAMILIES = (
    "white-hot rainbow aura with hard circular pressure rings",
    "pink-gold armor light with sharp prism sparks",
    "dragon-shaped aura coils with violent lightning arcs",
    "heart-star core light with strong beam recoil",
    "moon-prism energy with silver-blue impact rings",
    "phoenix-flame silhouette with blazing feather trails",
    "aurora bands with high-contrast final-form flashes",
    "faceted crystal light with explosive rainbow refractions",
    "royal sigil light with forceful crown-shaped shock rings",
    "shadow energy converting into clean rainbow flame",
    "candy-colored plasma with hot white energy cores",
    "celestial star-map light with piercing comet streaks",
)

RECENT_PHRASE_CANDIDATES = (
    "power stance",
    "battle stance",
    "dynamic pose",
    "dramatic power-up",
    "charged posture",
    "braces in",
    "braces low",
    "plants her feet",
    "planting her feet",
    "planted feet",
    "thrusting both hands upward",
    "thrusts both hands upward",
    "thrusting her wand forward",
    "thrusts her wand forward",
    "channels energy",
    "channels heart-star energy",
    "summons rainbow fire wings",
    "rainbow fire wings",
    "prismatic lightning",
    "effervescent rainbow energy",
    "glitter armor plates",
    "moon-prism aura",
    "silver-blue spell rings",
    "rippling circular shock rings",
    "celestial ribbon trails",
    "faceted rainbow refractions",
    "around her crown",
    "gold star flare",
    "golden light",
    "hair streams outward",
    "raises her visible hands",
    "radiates from her crown",
    "crackles around",
    "surges around",
    "erupts from her crown",
    "beam cannon",
    "castle tower",
    "city skyline",
    "neon cityscape",
    "tilts her head",
    "loose hair streams",
    "loose hair lifts",
    "loose hair flows",
    "long hair streams",
    "long hair flows",
    "raises her hand",
    "raises her right arm",
    "holds her staff",
    "expands from the crown",
    "expands from the staff tip",
    "brightens from the tip",
)

STYLE_FORBIDDEN_PHRASES = {
    "little-queen": (
        "contained spin",
        "dress billows",
        "dress fabric ripples",
        "executes a contained spin",
        "sparkling starlight bursts emanate from her crown",
        "sparkling starlight bursts emanate from her golden crown",
        "performs a contained spin",
        "performs a ribbon-dance sway",
        "pose pulse",
        "ribbon-dance sway",
        "skirt flares",
        "cute pose pulse",
        "performs a cute pose pulse",
        "executes a cute pose pulse",
        "cute pose pulse as her dress fabric ripples",
        "bounces in a storybook dance pose as her dress fabric ripples",
        "wand energy bursts",
        "wand energy spirals",
        "tilts her head",
        "holds her current pose",
        "stands as",
        "smiles as",
        "skirt billows",
        "head tilt",
        "gentle glow",
        "quiet pose",
        "simple glow",
        "barely",
        "faint",
        "subtle",
    ),
}

FORBIDDEN_CAMERA_MOTION = re.compile(
    r"\b(?:camera|pan(?:s|ned|ning)?|zoom(?:s|ed|ing)?|dolly|trucking)\b"
    r"|\b(?:push[- ]?in|pull[- ]?back|rack focus)\b"
    r"|\b(?:view|frame|framing)\s+(?:moves?|shifts?|drifts?)\b",
    flags=re.IGNORECASE,
)

FORBIDDEN_LARGE_MOTION = re.compile(
    r"\b(?:walk(?:s|ed|ing)?|run(?:s|ning)?|turn(?:s|ed|ing)?\s+away|"
    r"step(?:s|ped|ping)?|jump(?:s|ed|ing)?|leap(?:s|ed|ing)?|"
    r"lunge(?:s|d|ing)?|dash(?:es|ed|ing)?|spin(?:s|ning)?|"
    r"twirl(?:s|ed|ing)?)\b"
    r"|\bshift(?:s|ed|ing)?\s+(?:his|her|their|its)?\s*weight\b",
    flags=re.IGNORECASE,
)

OUTPUT_META_LANGUAGE = re.compile(
    r"\b(?:visible|shown|pictured|image)\b",
    flags=re.IGNORECASE,
)
BUILDUP_PHASE = re.compile(
    r"\b(?:charges?|compresses?|condenses?|gathers?|traces?|scans?|channels?|"
    r"draws?|tightens?|coils?|spirals?|aligns?|assembles?|wraps?|builds?|"
    r"pulses?|converges?|forms?|drains?)\b",
    flags=re.IGNORECASE,
)
AGGRESSIVE_PAYOFF = re.compile(
    r"\b(?:erupts?|explodes?|locks?|snaps?|unfolds?|fires?|releases?|launches?|"
    r"bursts?|blasts?|detonates?|slams?|resolves?|discharges?|roars?)\b",
    flags=re.IGNORECASE,
)
TWO_STAGE_CONNECTOR = re.compile(r"\bthen\b", flags=re.IGNORECASE)
UNSAFE_PLURAL_LIMBS = re.compile(
    r"\b(?:arms|palms|feet|legs)\b",
    flags=re.IGNORECASE,
)
HANDS_LANGUAGE = re.compile(r"\bhands\b", flags=re.IGNORECASE)
EXACT_BOTH_HANDS = re.compile(r"\bboth hands\b", flags=re.IGNORECASE)
HAIR_OR_FABRIC_REACTION = re.compile(
    r"\b(?:hair|fabric|skirt|cape|ribbon|dress)\b.{0,28}"
    r"\b(?:streams?|flows?|lifts?|surges?|snaps?|whips?|billows?|flares?)\b",
    flags=re.IGNORECASE,
)
LEAD_ACTION = re.compile(
    r"\b(?:transforms?|braces?|channels?|compresses?|summons?|ignites?|aims?|"
    r"releases?|raises?|holds?|extends?|grips?|lifts?|draws?|plants?|locks?|"
    r"unleashes?|charges?|traces?|gathers?|condenses?)\b",
    flags=re.IGNORECASE,
)
MULTIPLE_SUBJECT_LANGUAGE = re.compile(
    r"\b(?:large|small|smaller|main|second)\s+(?:subject|figure|girl|queen)\b"
    r"|\b(?:two|both)\s+(?:subjects|figures|girls|queens)\b",
    flags=re.IGNORECASE,
)
LITTLE_QUEEN_MODE_PATTERNS = {
    "transformation": (
        re.compile(
            r"\b(?:armor|armour|exosuit|final[- ]form|battle[- ]form|"
            r"energy wings?|aura wings?|mantle|plates?)\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:locks?|unfolds?|resolves?|snaps?|assembles?)\b",
            flags=re.IGNORECASE,
        ),
    ),
    "power-up": (
        re.compile(
            r"\b(?:aura|energy column|power[- ]?up|overload|reactor|"
            r"supernova|power state)\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:erupts?|detonates?|roars?|surges?|blasts?)\b",
            flags=re.IGNORECASE,
        ),
    ),
    "attack": (
        re.compile(
            r"\b(?:beam|blast|cannon|projectile|slash|lance|counterwave|"
            r"shockwave|shield|barrier|finisher)\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:fires?|launches?|releases?|slams?|blasts?)\b",
            flags=re.IGNORECASE,
        ),
    ),
    "environmental-spell": (
        re.compile(
            r"(?=.*\b(?:background|air|scene|structure|flowers?|vines?|branches|"
            r"crystals?|portal|ornaments?|lights?)\b)(?=.*\b(?:spell|sigil|"
            r"runes?|resonance)\b)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:bursts?|erupts?|discharges?|blasts?|releases?)\b",
            flags=re.IGNORECASE,
        ),
    ),
}

LTX_PROMPT_PREFIX = (
    "The camera remains locked with unchanged framing. Preserve the original "
    "subject's identity, appearance, clothing, location, and background while "
    "the described contained subject motion happens clearly. "
)
LTX_PROMPT_SUFFIX = (
    " Make the contained body, hair, fabric, lighting, and atmospheric "
    "effects clearly perceptible throughout the shot; no camera movement, "
    "reframing, new subjects, or scene change."
)
LTX_LITTLE_QUEEN_PREFIX = (
    "The camera remains locked with unchanged framing. Begin from the exact "
    "starting pose and keep every subject anchored in its source position. Preserve every "
    "face, outfit, physical prop, and background detail while the described "
    "high-energy motion happens clearly. "
)
LTX_LITTLE_QUEEN_SUFFIX = (
    " Only the described luminous transformation, aura, or attack overlays may "
    "materialize; no other new objects, identity or anatomy change, physical "
    "clothing replacement, camera movement, reframing, or scene change."
)


def build_ltx_prompt(motion_prompt: str, prompt_style: str = "rave") -> str:
    validate_motion_prompt(motion_prompt)
    if prompt_style == "little-queen":
        validate_style_prompt(motion_prompt, prompt_style)
        return f"{LTX_LITTLE_QUEEN_PREFIX}{motion_prompt}{LTX_LITTLE_QUEEN_SUFFIX}"
    return f"{LTX_PROMPT_PREFIX}{motion_prompt}{LTX_PROMPT_SUFFIX}"


def little_queen_mode_for_clip(clip_index: int) -> str:
    return LITTLE_QUEEN_MODE_SCHEDULE[clip_index % len(LITTLE_QUEEN_MODE_SCHEDULE)]


def fallback_prompt_for_style(prompt_style: str, clip_index: int = 0) -> str:
    if prompt_style == "little-queen":
        return LITTLE_QUEEN_MODE_FALLBACKS[little_queen_mode_for_clip(clip_index)]
    return FALLBACK_PROMPTS.get(prompt_style, FALLBACK_PROMPT)


def little_queen_allows_secondary_reaction(clip_index: int) -> bool:
    return clip_index % 5 == 4


def parse_little_queen_response(text: str) -> tuple[str, bool]:
    answer = strip_reasoning(text).strip()
    evidence = re.search(
        r"\bTWO_HANDS_CLEAR\s*:\s*(yes|no)\b",
        answer,
        flags=re.IGNORECASE,
    )
    prompt_match = re.search(
        r"\bPROMPT\s*:\s*(.+)",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )
    prompt = clean_prompt(prompt_match.group(1) if prompt_match else answer)
    return prompt, bool(evidence and evidence.group(1).lower() == "yes")


def _lead_action(prompt: str) -> str | None:
    match = LEAD_ACTION.search(prompt)
    return match.group(0).lower() if match else None


def _normalized_prompt(prompt: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", prompt.lower()))


def validate_little_queen_mode_prompt(
    prompt: str,
    mode: str,
    recent_prompts: list[str] | None = None,
    *,
    clip_index: int | None = None,
    two_hands_clear: bool = False,
) -> None:
    if mode not in LITTLE_QUEEN_MODE_PATTERNS:
        raise ValueError(f"Unknown Little Queen prompt mode: {mode}")
    if MULTIPLE_SUBJECT_LANGUAGE.search(prompt):
        raise RuntimeError(
            f"Little Queen prompt describes multiple rendered subjects: {prompt!r}"
        )
    if UNSAFE_PLURAL_LIMBS.search(prompt):
        raise RuntimeError(
            f"Little Queen prompt contains unsafe plural anatomy: {prompt!r}"
        )
    if HANDS_LANGUAGE.search(prompt):
        remaining = EXACT_BOTH_HANDS.sub("", prompt)
        if HANDS_LANGUAGE.search(remaining) or not two_hands_clear:
            raise RuntimeError(
                "Little Queen prompt uses plural hands without verified two-hand "
                f"image evidence: {prompt!r}"
            )

    connector = TWO_STAGE_CONNECTOR.search(prompt)
    if connector is None:
        raise RuntimeError(
            f"Little Queen prompt lacks an explicit buildup-to-payoff connector: {prompt!r}"
        )
    buildup = prompt[: connector.start()]
    payoff = prompt[connector.end() :]
    if not BUILDUP_PHASE.search(buildup):
        raise RuntimeError(
            f"Little Queen prompt lacks an energy buildup before 'then': {prompt!r}"
        )
    if not AGGRESSIVE_PAYOFF.search(payoff):
        raise RuntimeError(
            f"Little Queen prompt lacks an aggressive payoff after 'then': {prompt!r}"
        )

    outcome_pattern, mode_payoff_pattern = LITTLE_QUEEN_MODE_PATTERNS[mode]
    if not outcome_pattern.search(prompt) or not mode_payoff_pattern.search(payoff):
        raise RuntimeError(
            f"Little Queen prompt does not satisfy required {mode} mode: {prompt!r}"
        )

    if (
        clip_index is not None
        and HAIR_OR_FABRIC_REACTION.search(prompt)
        and not little_queen_allows_secondary_reaction(clip_index)
    ):
        raise RuntimeError(
            "Little Queen prompt uses a hair/fabric reaction in a slot reserved "
            f"for primary action only: {prompt!r}"
        )

    recent = recent_prompts or []
    lead_action = _lead_action(prompt)
    if lead_action is not None:
        recent_leads = [_lead_action(item) for item in recent[-8:]]
        if recent_leads.count(lead_action) >= 2:
            raise RuntimeError(
                f"Little Queen prompt repeats recent lead action {lead_action!r}: {prompt!r}"
            )
    if HAIR_OR_FABRIC_REACTION.search(prompt) and sum(
        bool(HAIR_OR_FABRIC_REACTION.search(item)) for item in recent[-4:]
    ) >= 2:
        raise RuntimeError(
            f"Little Queen prompt repeats recent hair/fabric choreography: {prompt!r}"
        )

    normalized = _normalized_prompt(prompt)
    for prior in recent[-8:]:
        similarity = SequenceMatcher(
            None,
            normalized,
            _normalized_prompt(prior),
        ).ratio()
        if similarity >= 0.82:
            raise RuntimeError(
                "Little Queen prompt is structurally too similar to a recent "
                f"prompt ({similarity:.0%}): {prompt!r}"
            )


class GemmaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        start_command: Path | None = None,
        timeout: float = 180.0,
        prompt_style: str = "rave",
    ) -> None:
        if prompt_style not in SYSTEM_PROMPTS:
            raise ValueError(f"Unknown Gemma prompt style: {prompt_style}")
        self.base_url = base_url.rstrip("/")
        self.start_command = start_command
        self.timeout = timeout
        self.prompt_style = prompt_style
        self.started_server = False

    def ensure_ready(self) -> None:
        if self._healthy():
            return
        if self.start_command is None:
            raise RuntimeError(f"Gemma server is not reachable at {self.base_url}")
        subprocess.run([str(self.start_command)], check=True)
        self.started_server = True
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self._healthy():
                return
            time.sleep(1)
        raise RuntimeError(f"Gemma server did not become ready at {self.base_url}")

    def describe_motion(
        self,
        image_path: Path,
        *,
        clip_index: int = 0,
        recent_prompts: list[str] | None = None,
    ) -> str:
        mime_type, image_bytes = prepare_image(
            image_path,
            max_edge=512 if self.prompt_style == "little-queen" else 384,
        )
        encoded = base64.b64encode(image_bytes).decode("ascii")
        correction = ""
        last_error: RuntimeError | None = None
        recent = recent_prompts or []
        mode = (
            little_queen_mode_for_clip(clip_index)
            if self.prompt_style == "little-queen"
            else None
        )
        for _ in range(4):
            prompt_text = SYSTEM_PROMPTS[self.prompt_style]
            prompt_text += build_diversity_instruction(
                self.prompt_style,
                clip_index,
                recent,
            )
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt_text + correction,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{encoded}"
                                },
                            },
                        ],
                    }
                ],
                "temperature": 0.40 if self.prompt_style == "little-queen" else 0.25,
                "top_p": 0.96 if self.prompt_style == "little-queen" else 0.9,
                "top_k": 60 if self.prompt_style == "little-queen" else 40,
                "max_tokens": 256,
            }
            request = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
            try:
                text = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"Unexpected Gemma response: {body}") from exc
            raw_answer = strip_reasoning(text)
            try:
                if self.prompt_style == "little-queen":
                    prompt, two_hands_clear = parse_little_queen_response(raw_answer)
                else:
                    prompt = clean_prompt(raw_answer)
                    two_hands_clear = False
                validate_motion_prompt(prompt)
                validate_style_prompt(prompt, self.prompt_style)
                if mode is not None:
                    validate_little_queen_mode_prompt(
                        prompt,
                        mode,
                        recent,
                        clip_index=clip_index,
                        two_hands_clear=two_hands_clear,
                    )
                return prompt
            except RuntimeError as exc:
                last_error = exc
                correction = (
                    f"\nYour previous answer {raw_answer!r} failed validation: {exc}. "
                    "Rewrite it without "
                    "camera language, spatial travel, or a scene change. Keep the "
                    "subject in place and use only shown anatomy, physical props, "
                    "and setting details. "
                    f"The required mode remains {mode}; do not downgrade it to a "
                    "head tilt, hair motion, quiet pose, or simple glow. Write one "
                    "coherent charge-to-payoff arc that ends in an eruption, "
                    "release, transformation lock, wing unfold, beam, or "
                    "shockwave. Follow the required TWO_HANDS_CLEAR/PROMPT output "
                    "format. Hair and fabric are optional. Say both hands only "
                    "when two distinct hands are clearly present."
                )
        raise last_error or RuntimeError("Gemma did not return a usable prompt")

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/health", timeout=2
            ) as response:
                data = json.load(response)
            return data.get("status") == "ok"
        except (OSError, ValueError, urllib.error.URLError):
            return False


def build_diversity_instruction(
    prompt_style: str,
    clip_index: int,
    recent_prompts: list[str],
) -> str:
    if prompt_style != "little-queen":
        return ""

    mode = little_queen_mode_for_clip(clip_index)
    mode_families = LITTLE_QUEEN_MODE_FAMILIES[mode]
    move_family = mode_families[clip_index % len(mode_families)]
    effect_family = LITTLE_QUEEN_V7_EFFECT_FAMILIES[
        (clip_index * 7 + clip_index // len(LITTLE_QUEEN_MODE_SCHEDULE))
        % len(LITTLE_QUEEN_V7_EFFECT_FAMILIES)
    ]
    recent_bans = recent_phrase_bans(recent_prompts)
    allow_secondary_reaction = little_queen_allows_secondary_reaction(clip_index)
    ban_text = ""
    if recent_bans:
        ban_text = (
            "\nAvoid these recently overused phrases or close variants: "
            + ", ".join(f'"{phrase}"' for phrase in recent_bans)
            + "."
        )

    return (
        "\n\nRequired creative assignment:"
        f"\nMode: {mode}. {LITTLE_QUEEN_MODE_DIRECTIVES[mode]}"
        f"\nChoreography theme: {move_family}."
        f"\nEffect palette: {effect_family}."
        "\nThe mode and high-energy payoff are mandatory. Adapt the physical "
        "anchor to the crop and pose, but never replace the assignment with "
        "gentle motion. A luminous armor, gauntlet, wing, mantle, aura, or beam "
        "is an allowed temporary overlay, not a physical-object invention."
        "\nUse one evolving effect system with an energy buildup, the word "
        "'then', and a forceful mode-specific payoff. "
        + (
            "A brief hair or fabric reaction is allowed in this slot, but it "
            "must remain secondary."
            if allow_secondary_reaction
            else "Do not mention hair or fabric motion in this slot."
        )
        + " If a prop or background detail is uncertain, use the silhouette, "
        "existing background light, or air rather than naming an object."
        f"{ban_text}"
    )


def recent_phrase_bans(recent_prompts: list[str], limit: int = 12) -> list[str]:
    recent_text = " ".join(recent_prompts[-8:]).lower()
    bans = [
        phrase
        for phrase in RECENT_PHRASE_CANDIDATES
        if phrase in recent_text
    ]
    return bans[:limit]


def clean_prompt(text: str) -> str:
    text = strip_reasoning(text)
    prompt = " ".join(text.strip().split())
    prompt = re.sub(
        r"^(?:prompt|image-to-video prompt|video prompt)\s*:\s*",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    return prompt.strip(" \"'`")


def strip_reasoning(text: str) -> str:
    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def validate_motion_prompt(prompt: str) -> None:
    word_count = len(prompt.split())
    if word_count < 12 or word_count > 48:
        raise RuntimeError(
            f"Gemma prompt must contain 12-48 words, got {word_count}: {prompt!r}"
        )
    if FORBIDDEN_CAMERA_MOTION.search(prompt):
        raise RuntimeError(f"Gemma prompt contains camera motion: {prompt!r}")
    if FORBIDDEN_LARGE_MOTION.search(prompt):
        raise RuntimeError(f"Gemma prompt contains large body motion: {prompt!r}")


def validate_style_prompt(prompt: str, prompt_style: str) -> None:
    lowered = prompt.lower()
    for phrase in STYLE_FORBIDDEN_PHRASES.get(prompt_style, ()):
        if phrase in lowered:
            raise RuntimeError(
                f"Gemma prompt is too generic for {prompt_style}: {prompt!r}"
            )
    if prompt_style == "little-queen":
        word_count = len(prompt.split())
        if word_count < 20 or word_count > 40:
            raise RuntimeError(
                "Little Queen prompt must contain 20-40 words, "
                f"got {word_count}: {prompt!r}"
            )
        if UNSAFE_PLURAL_LIMBS.search(prompt):
            raise RuntimeError(
                f"Little Queen prompt contains unsafe plural anatomy: {prompt!r}"
            )
        if OUTPUT_META_LANGUAGE.search(prompt):
            raise RuntimeError(
                f"Little Queen prompt contains visual-analysis language: {prompt!r}"
            )


def prepare_image(path: Path, max_edge: int = 384) -> tuple[str, bytes]:
    guessed_type = mimetypes.guess_type(path.name)[0] or "image/png"
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
    mime_type = "image/jpeg" if guessed_type.startswith("image/") else guessed_type
    return mime_type, buffer.getvalue()
