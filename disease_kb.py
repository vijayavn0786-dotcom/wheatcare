"""
disease_kb.py - Comprehensive Wheat Pathology Knowledge Base
Contains scientific descriptions, epidemiology, symptoms, cultural controls,
organic remedies, chemical fungicides, and severity classification rules.
"""

DISEASE_KNOWLEDGE_BASE = {
    "Brown rust": {
        "display_name": "Brown Rust (Leaf Rust)",
        "scientific_name": "Puccinia triticina",
        "category": "Fungal Basidiomycete",
        "primary_hosts": "Wheat (Triticum aestivum), Durum wheat, Triticale",
        "severity_grade": {
            "mild": "Low lesion density (<15% leaf surface area). Early vegetative stage infection.",
            "moderate": "Dispersed pustules (15%–35% leaf surface area). Potential 10%–20% yield drag.",
            "severe": "Dense coalescing pustules (>35% leaf area). Flag leaf compromised; up to 50% yield loss.",
        },
        "optimal_conditions": {
            "temperature": "15°C to 25°C (59°F to 77°F)",
            "humidity": "Relative humidity > 85%, free moisture on leaf for 4–8 hours",
            "season": "Mid-to-late crop season, warm sunny days and cool dewy nights",
        },
        "symptoms": [
            "Small, circular to oval, reddish-brown pustules (uredinia) scattered randomly on upper leaf surfaces.",
            "Pustules rupture the epidermis to release powdery orange-brown spores upon contact.",
            "Chlorotic (yellowish) halo frequently surrounds individual pustules as infection advances.",
            "In severe late-stage infections, pustules turn dark brown/black (telia stage) as plant matures.",
        ],
        "cultural_control": [
            "Select and sow certified rust-resistant wheat cultivars adapted to regional pathotypes.",
            "Eradicate volunteer wheat plants and alternate hosts (e.g., Thalictrum species) during fallow.",
            "Avoid excessive early nitrogen application which produces overly lush, susceptible canopy tissue.",
            "Adopt recommended row spacing to improve canopy ventilation and reduce leaf wetness duration.",
        ],
        "organic_remedies": [
            "Foliar applications of bio-fungicides containing *Bacillus subtilis* or *Trichoderma harzianum*.",
            "Dilute neem oil formulation (0.5%–1.0%) or potassium silicate to reinforce epidermal cell walls.",
            "Compost tea sprays to enhance competing phyllosphere microflora in low-pressure settings.",
        ],
        "chemical_control": [
            "Triazole fungicides: Tebuconazole (250 EC @ 1.0 ml/L) or Propiconazole (25 EC @ 1.0 ml/L).",
            "Strobilurin + Triazole premixes: Azoxystrobin + Difenoconazole for dual protective & curative action.",
            "Timing: Apply at first detection on lower canopy or prior to heading when flag leaf emerges.",
        ],
        "economic_threshold": "Spray when 1%–5% of lower leaves or any upper canopy leaves show active uredinia.",
    },
    "Yellow rust": {
        "display_name": "Yellow Rust (Stripe Rust)",
        "scientific_name": "Puccinia striiformis f. sp. tritici",
        "category": "Fungal Basidiomycete",
        "primary_hosts": "Bread wheat, Durum wheat, Barley, Rye, wild grasses",
        "severity_grade": {
            "mild": "Sparse yellow stripes on lower leaves (<15% leaf area).",
            "moderate": "Linear stripe progression reaching the second leaf (15%–35% area).",
            "severe": "Extensive systemic leaf necrosis (>35% area). Flag leaf and spikes infected; catastrophic yield loss possible (up to 70%).",
        },
        "optimal_conditions": {
            "temperature": "10°C to 15°C (50°F to 59°F), inhibited above 22°C",
            "humidity": "High humidity, prolonged fog, dewy conditions, cool damp climates",
            "season": "Early to mid-spring; prevalent in high-latitude or highland wheat zones",
        },
        "symptoms": [
            "Bright lemon-yellow to orange-yellow pustules arranged in parallel, linear stripes along leaf veins.",
            "In young seedlings, pustules may appear scattered before resolving into distinctive stripes on adult leaves.",
            "Infected leaves experience premature desiccation, curling, and brown necrotic striping.",
            "In severe epidemics, glumes and awns inside wheat spikes are invaded, shrivelling developing grain.",
        ],
        "cultural_control": [
            "Deploy multi-gene resistant wheat cultivars (APR - Adult Plant Resistance genes like Yr18/Lr34).",
            "Synchronize sowing within regional optimal planting windows to escape peak spore showers.",
            "Maintain balanced fertilization with sufficient potassium and phosphorus to enhance tissue resistance.",
            "Deep plow crop stubble post-harvest to disrupt localized green bridges.",
        ],
        "organic_remedies": [
            "Preventive copper hydroxide or copper oxychloride at early seedling stages under organic standards.",
            "Biological antagonism using *Pseudomonas fluorescens* bio-formulations.",
            "Sulfur-based dusts or wettable sulfur applied strictly before stripe proliferation occurs.",
        ],
        "chemical_control": [
            "Systemic fungicides: Propiconazole 25% EC (500 ml/ha) or Tebuconazole 25% WG (750 g/ha).",
            "Dual-action combinations: Epoxiconazole + Pyraclostrobin or Metconazole + Fluxapyroxad.",
            "Timing: Extremely critical; act immediately upon detecting the initial stripe foci ('hotspots').",
        ],
        "economic_threshold": "Immediate spot or field treatment upon detecting active stripe rust foci in cool weather.",
    },
    "Loose Smut": {
        "display_name": "Loose Smut",
        "scientific_name": "Ustilago tritici",
        "category": "Internally Seed-Borne Fungal Basidiomycete",
        "primary_hosts": "Wheat, Rye",
        "severity_grade": {
            "mild": "Occasional smutted head (<2% field incidence).",
            "moderate": "Localized infected clusters (2%–5% incidence).",
            "severe": "Widespread blackened spikes (>5% incidence). Significant direct yield reduction.",
        },
        "optimal_conditions": {
            "temperature": "16°C to 22°C (61°F to 72°F) during anthesis (flowering)",
            "humidity": "Frequent rain, damp cloudy weather during heading facilitates floral infection",
            "season": "Infection occurs during flowering; symptoms express at heading next generation",
        },
        "symptoms": [
            "Normal grain spikelets completely replaced by a powdery mass of olive-black fungal spores (teliospores).",
            "Infected heads typically emerge earlier than healthy heads and stand erect on the upper canopy.",
            "The delicate silvery enclosing membrane ruptures quickly, leaving only a bare, naked rachis.",
            "Wind blows teliospores to adjacent flowering florets, establishing dormant mycelium inside embryo.",
        ],
        "cultural_control": [
            "Strictly plant certified, laboratory-tested disease-free seed stocks.",
            "Use solar heat seed treatment: Soak seeds in water for 4–5 hours in summer, then spread under hot sun for 4 hours.",
            "Hot-water seed treatment: Presoak at 20°C for 5 hours, then immerse in 52°C water for exactly 11 minutes.",
            "Rogue out and destroy infected smutted heads in sealed plastic bags before spore dispersion begins.",
        ],
        "organic_remedies": [
            "Physical hot-water thermotherapy as certified organic seed treatment.",
            "Seed priming with beneficial bio-agents (*Trichoderma viride* @ 4 g/kg seed).",
            "Cultivar resistance selection for high floral resistance.",
        ],
        "chemical_control": [
            "Foliar fungicides CANNOT cure loose smut once symptoms appear on the spike.",
            "MANDATORY Seed Treatment: Carboxin 37.5% + Thiram 37.5% DS (@ 2.5–3.0 g/kg seed).",
            "Alternative systemic seed dressings: Tebuconazole 2% DS (@ 1.0–1.2 g/kg seed) or Difenoconazole.",
        ],
        "economic_threshold": "Zero-tolerance for foundation and certified seed production fields.",
    },
    "Septoria": {
        "display_name": "Septoria Leaf Blotch",
        "scientific_name": "Zymoseptoria tritici (anamorph Septoria tritici)",
        "category": "Fungal Ascomycete (Hemibiotroph)",
        "primary_hosts": "Durum and Bread Wheat",
        "severity_grade": {
            "mild": "Small chlorotic flecks or isolated blotches on lower foliage (<15% area).",
            "moderate": "Lesions merging on middle canopy leaves with visible pycnidia specks (15%–35% area).",
            "severe": "Extensive brown necrotic blighting across flag leaf and ear (>35% area); severe canopy collapse.",
        },
        "optimal_conditions": {
            "temperature": "15°C to 20°C (59°F to 68°F)",
            "humidity": "Rain-splash dispersal; requires prolonged leaf wetness (>24–48 hours)",
            "season": "Prolonged wet springs, coastal climates, intensive rain-fed cultivation",
        },
        "symptoms": [
            "Irregular, elongated or oval chlorotic spots that develop into tan, straw-coloured necrotic lesions.",
            "Distinctive diagnostic hallmark: Tiny, black pepper-like fruiting bodies (pycnidia) embedded inside lesions.",
            "Lesions expand parallel to leaf veins, frequently coalescing into extensive necrotic blotches.",
            "Causes premature senescence of photosynthetic leaf tissue, leading to severely shrivelled kernels.",
        ],
        "cultural_control": [
            "Rotate crops for at least 1–2 seasons with non-host crops (oilseed rape, pulses, maize).",
            "Shred and deeply incorporate infected crop residues to accelerate microbial decomposition of pycnidia.",
            "Avoid overly dense plant populations and late planting to minimize splash canopy transmission.",
            "Cultivate varieties with proven Stb gene resistance packages.",
        ],
        "organic_remedies": [
            "Bordeaux mixture or copper octanoate applied protectively prior to prolonged rain events.",
            "Foliar biocontrol applications of *Bacillus amyloliquefaciens* formulations.",
            "Silicon soil amendments to enhance mechanical plant cell wall toughness against fungal penetration.",
        ],
        "chemical_control": [
            "Multi-site protectant: Folpet or Chlorothalonil (where permitted) to mitigate resistance buildup.",
            "SDHI (Succinate Dehydrogenase Inhibitors) + Azole mixtures: Bixafen + Prothioconazole.",
            "QoI/Strobilurin fungicides: Only effective if local populations are not resistant to G143A mutation.",
            "Optimal Timing: GS 32 to GS 39 (emergence of final leaf three up to flag leaf emergence).",
        ],
        "economic_threshold": "Fungicide application recommended when pycnidia are confirmed on the third leaf from top at stem elongation.",
    },
    "Healthy": {
        "display_name": "Healthy Wheat Foliage",
        "scientific_name": "Triticum aestivum",
        "category": "Non-pathological / Physiological Normal State",
        "primary_hosts": "Agricultural staple cereal crop",
        "severity_grade": {
            "mild": "Not applicable — specimen is vigorous and healthy.",
            "moderate": "Not applicable — maintain standard surveillance.",
            "severe": "Not applicable.",
        },
        "optimal_conditions": {
            "temperature": "15°C to 24°C daytime, cooler nights",
            "humidity": "Moderate atmospheric humidity (50%–70%), well-drained loamy soil",
            "season": "Optimal vegetative growth with balanced solar irradiance and soil nutrition",
        },
        "symptoms": [
            "Uniform vibrant green leaf color across the leaf blade and sheath.",
            "Smooth epidermal surface devoid of powdery pustules, chlorotic halos, or pycnidial speckling.",
            "Normal turgidity, healthy leaf margins, and unblemished ligules and auricles.",
            "Strong photosynthetic capacity supporting vigorous tillering and grain fill.",
        ],
        "cultural_control": [
            "Continue systematic crop scouting across fields on a 7-day scouting interval.",
            "Ensure precision fertilizer scheduling based on split nitrogen soil test recommendations.",
            "Maintain optimal irrigation schedules to prevent drought stress or prolonged canopy waterlogging.",
            "Conduct routine weed management to prevent grass weeds from serving as cryptic fungal reservoirs.",
        ],
        "organic_remedies": [
            "Maintain soil organic carbon via compost or green manuring to nurture beneficial soil microbiome.",
            "Inoculate soil with mycorrhizal fungi to enhance nutrient and water absorption efficiency.",
            "Apply sea-kelp extract or humic acids as foliar biostimulants to sustain cellular vigor.",
        ],
        "chemical_control": [
            "No chemical intervention or pesticide application required.",
            "Preserve natural beneficial insect predators and protective phyllosphere microflora.",
            "Do not apply preventive fungicides unnecessarily to prevent environmental burden and resistance.",
        ],
        "economic_threshold": "Zero action threshold; continue routine visual monitoring.",
    },
}
