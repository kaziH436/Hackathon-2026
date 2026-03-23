"""
generate_training_data.py
Produces nlp/data/policy_labels.jsonl with 50,000 labelled training examples.
10,000 examples per category x 5 categories = 50,000 total.

Complex specialised domain — California urban digital twin simulating
traffic, pollution, and energy across real sensor infrastructure.
10,000 examples per category is appropriate for this domain complexity.

Usage:
    python nlp/data/generate_training_data.py [--output-dir PATH]

Output:
    nlp/data/policy_labels.jsonl  — one JSON record per line: {"text": str, "label": str}

Security & error handling:
  #1  Path traversal prevention — output path validated against allowed base.
  #2  Unbounded makedirs prevention — makedirs only after path validated.
  #3  Atomic write — .tmp file swapped in via os.replace() on success;
      cleaned up on failure. Mid-write crash never corrupts existing file.
  #4  Output size ceiling — hard cap via MAX_EXAMPLES_PER_CATEGORY.
  #5  Explicit raises not assert — assert stripped under Python -O flag.
  #6  Slot key validation — all template slot references validated against
      SLOTS at startup before any generation begins.
  #7  Duplicate slot value detection — duplicates removed from each slot
      list at startup with a warning so pool sizes are accurate.
  #8  Cartesian product memory guard — templates with >500k combinations
      are capped to random sampling to prevent RAM exhaustion.
  #9  Duplicate example detection — exact duplicate sentences tracked and
      warned about so effective dataset size is always known.
  #10 Robust tmp path — explicit string suffix rather than with_suffix()
      to avoid fragile behaviour on non-standard filenames.
"""

import argparse
import json
import random
import re
import os
import sys
from pathlib import Path
from itertools import product as cartesian_product
from collections import Counter

# ── Configuration ────────────────────────────────────────────────────────────

SEED = 42
EXAMPLES_PER_CATEGORY = 10_000   # 10,000 x 5 categories = 50,000 total

# Security fix #4: hard ceiling to prevent accidental runaway output
MAX_EXAMPLES_PER_CATEGORY = 100_000

# Security fix #8: max cartesian product size per template before
# falling back to random sampling to prevent RAM exhaustion
MAX_CARTESIAN_PER_TEMPLATE = 500_000

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILENAME = "policy_labels.jsonl"


def _resolve_output_path(output_dir: str | None) -> Path:
    """
    Return a validated, absolute output path.

    Security fix #1 & #2 — path traversal + unbounded makedirs:
    Resolves and validates any --output-dir argument against the allowed
    base directory, blocking inputs like '../../etc'.
    """
    if output_dir is None:
        base = _DEFAULT_OUTPUT_DIR
    else:
        candidate = Path(output_dir).resolve()
        allowed_base = _DEFAULT_OUTPUT_DIR.resolve()
        try:
            candidate.relative_to(allowed_base)
        except ValueError:
            raise ValueError(
                f"Output directory '{candidate}' is outside the allowed base "
                f"'{allowed_base}'. Refusing to write there."
            )
        base = candidate

    return base / OUTPUT_FILENAME


def _validate_slots(templates: dict, slots: dict) -> None:
    """
    Security fix #6 — slot key validation:
    Verify every {slot} reference in every template exists in SLOTS.
    Raises ValueError at startup before any generation begins,
    so a typo is caught immediately rather than mid-generation.
    """
    missing = []
    for category, tmpl_list in templates.items():
        for tmpl in tmpl_list:
            for ph in re.findall(r"\{(\w+)\}", tmpl):
                if ph not in slots:
                    missing.append(
                        f"  Category '{category}': template references "
                        f"unknown slot '{{{ph}}}' — not found in SLOTS."
                    )
    if missing:
        raise ValueError(
            "Template slot validation failed:\n" + "\n".join(missing) +
            "\nFix the slot name(s) before re-running."
        )


def _deduplicate_slots(slots: dict) -> dict:
    """
    Security fix #7 — duplicate slot value detection:
    Remove duplicate values from each slot list and warn so pool size
    estimates are accurate. Duplicates silently inflate apparent diversity
    while reducing actual variety in generated sentences.
    """
    cleaned = {}
    for key, values in slots.items():
        seen = []
        dupes = []
        for v in values:
            if v not in seen:
                seen.append(v)
            else:
                dupes.append(v)
        if dupes:
            print(
                f"  ⚠  Slot '{key}': removed {len(dupes)} duplicate(s): "
                + ", ".join(f'"{d}"' for d in dupes),
                file=sys.stderr,
            )
        cleaned[key] = seen
    return cleaned


# ── Slot vocabularies ─────────────────────────────────────────────────────────
# California-specific geography. Duplicates are detected and removed at
# startup by _deduplicate_slots() — security fix #7.

SLOTS = {
    "road": [
        # Los Angeles surface streets
        "Sunset Blvd", "Santa Monica Blvd", "Wilshire Blvd", "Hollywood Blvd",
        "Ventura Blvd", "Sepulveda Blvd", "Lincoln Blvd", "Melrose Ave",
        "Cahuenga Blvd", "Figueroa St", "Vermont Ave", "Western Ave",
        "Crenshaw Blvd", "Alameda St", "Spring St", "Main St",
        "Venice Blvd", "Pico Blvd", "Olympic Blvd", "Washington Blvd",
        "Adams Blvd", "Jefferson Blvd", "Manchester Ave", "Florence Ave",
        "Century Blvd", "Imperial Hwy", "Rosecrans Ave", "El Segundo Blvd",
        "Hawthorne Blvd", "Aviation Blvd", "Inglewood Ave", "Prairie Ave",
        "La Cienega Blvd", "La Brea Ave", "Fairfax Ave", "Highland Ave",
        "Vine St", "Gower St", "Beachwood Dr", "Mulholland Dr",
        "Laurel Canyon Blvd", "Coldwater Canyon Ave", "Beverly Glen Blvd",
        "San Vicente Blvd", "Robertson Blvd", "Doheny Dr", "Crescent Dr",
        "Bundy Dr", "Barrington Ave", "Centinela Ave", "Sawtelle Blvd",
        "Overland Ave", "Motor Ave", "National Blvd", "Culver Blvd",
        # Los Angeles freeways
        "Interstate 405", "Interstate 10", "Interstate 110",
        "Highway 101", "Highway 1", "Highway 60", "Highway 91",
        "Interstate 210", "Interstate 605", "Interstate 710", "Interstate 105",
        "Interstate 405 northbound", "Interstate 405 southbound",
        "Interstate 10 eastbound", "Interstate 10 westbound",
        "the 110 freeway", "the 101 freeway", "the 405 freeway",
        # San Francisco surface streets
        "Market St SF", "Mission St", "Valencia St", "Castro St",
        "Geary Blvd", "Van Ness Ave", "19th Ave", "Lombard St",
        "Columbus Ave", "Embarcadero", "Folsom St",
        "Howard St", "Brannan St", "King St SF", "Berry St",
        "3rd St SF", "4th St SF", "7th St SF", "8th St SF",
        "Townsend St", "Cesar Chavez St", "Potrero Ave", "Guerrero St",
        "Dolores St", "Church St", "Sanchez St", "Noe St",
        "Divisadero St", "Fillmore St", "Webster St", "Buchanan St",
        "Laguna St", "Octavia Blvd", "Gough St", "Franklin St",
        "Oak St", "Fell St", "Page St", "Haight St",
        "Stanyan St", "Clayton St", "Cole St", "Carl St",
        "Irving St", "Judah St", "Noriega St", "Taraval St",
        # Bay Area freeways
        "US-101 Bay Area", "I-280", "I-880", "I-80 Bay Area", "I-580",
        "I-680", "SR-24", "SR-92", "SR-84", "SR-87",
        # San Diego surface streets
        "Harbor Dr SD", "Pacific Hwy SD", "El Cajon Blvd", "University Ave SD",
        "Balboa Ave", "Morena Blvd", "Garnet Ave", "Mission Blvd SD",
        "Sports Arena Blvd", "Rosecrans St SD", "Nimitz Blvd", "Midway Dr SD",
        "National Ave SD", "Imperial Ave SD",
        "Logan Ave", "Highland Ave SD", "Euclid Ave SD",
        # San Diego freeways
        "Interstate 8", "Interstate 15 SD", "Interstate 5 SD", "Interstate 805",
        "Interstate 163", "SR-94", "SR-52", "SR-56", "SR-78",
        # Sacramento surface streets
        "J St Sacramento", "K St Sacramento", "L St Sacramento",
        "16th St Sacramento", "21st St Sacramento",
        "Folsom Blvd Sacramento", "Sunrise Blvd", "Watt Ave", "Stockton Blvd",
        "Broadway Sacramento", "Del Paso Blvd", "Florin Rd", "Mack Rd",
        "Fruitridge Rd", "Elder Creek Rd", "Power Inn Rd", "Bradshaw Rd",
        # Sacramento freeways
        "Interstate 5 Sacramento", "Interstate 80 Sacramento", "US-50", "SR-99",
        "Business 80", "Capital City Freeway",
        # General California
        "Pacific Coast Highway", "State Route 1", "State Route 99 Central Valley",
        "State Route 58", "State Route 138", "State Route 14",
        "State Route 18", "State Route 38", "State Route 62",
    ],
    "area": [
        # Los Angeles neighbourhoods
        "Downtown LA", "Hollywood", "Santa Monica", "Venice Beach",
        "Culver City", "Koreatown", "Compton", "Inglewood",
        "Long Beach", "Pasadena", "Burbank", "Glendale",
        "the San Fernando Valley", "the Port of LA", "the LAX corridor",
        "West Hollywood", "Silver Lake", "Echo Park", "Los Feliz",
        "Boyle Heights", "East LA", "Watts", "Hawthorne",
        "Torrance", "Carson", "Gardena", "Redondo Beach",
        "Manhattan Beach", "El Segundo", "Playa Vista", "Mar Vista",
        "Palms", "Westwood", "Brentwood", "Pacific Palisades",
        "Malibu", "Topanga", "Reseda", "Van Nuys",
        "North Hollywood", "Studio City", "Sherman Oaks", "Encino",
        "Tarzana", "Woodland Hills", "Canoga Park", "Chatsworth",
        "Granada Hills", "Northridge", "Panorama City", "Sun Valley",
        "Sunland", "Tujunga", "La Canada", "Montrose",
        "Alhambra", "Monterey Park", "Rosemead", "El Monte",
        "Baldwin Park", "West Covina", "Covina", "Azusa",
        "Glendora", "San Dimas", "La Verne", "Pomona",
        "Ontario", "Rancho Cucamonga", "Fontana", "Rialto",
        "San Bernardino", "Colton", "Loma Linda", "Redlands",
        "Downey", "Norwalk", "Cerritos", "Lakewood",
        "Bellflower", "Paramount", "Lynwood", "South Gate",
        "Huntington Park", "Bell", "Maywood", "Bell Gardens",
        "Commerce", "Vernon", "Cudahy", "Walnut Park",
        # San Francisco neighbourhoods
        "San Francisco", "Downtown San Francisco", "the Financial District SF",
        "the Mission District", "SoMa", "Chinatown SF", "the Tenderloin",
        "the Castro", "Noe Valley", "the Richmond District",
        "the Sunset District", "Haight-Ashbury", "the Marina SF",
        "Pacific Heights", "Nob Hill", "Russian Hill", "North Beach SF",
        "Fishermans Wharf", "Potrero Hill", "Bernal Heights",
        "Glen Park", "Excelsior", "Visitacion Valley", "Bayview SF",
        "Hunters Point", "Dogpatch", "Mission Bay SF",
        # Bay Area cities
        "Oakland", "Berkeley", "San Jose", "Palo Alto",
        "Silicon Valley", "the Peninsula", "Marin County",
        "the East Bay", "Fremont", "Hayward", "Richmond CA",
        "San Leandro", "Castro Valley", "San Lorenzo", "Union City",
        "Newark CA", "Milpitas", "Santa Clara", "Sunnyvale",
        "Mountain View", "Los Altos", "Cupertino", "Saratoga CA",
        "Campbell CA", "Los Gatos", "Morgan Hill", "Gilroy",
        "San Mateo", "Foster City", "Belmont CA", "San Carlos",
        "Redwood City", "Menlo Park", "Atherton", "Portola Valley",
        "Woodside CA", "Half Moon Bay", "Pacifica", "Daly City",
        "South San Francisco", "Brisbane CA", "San Bruno", "Millbrae",
        "Burlingame", "Walnut Creek", "Concord CA", "Pleasant Hill CA",
        "Martinez CA", "Pittsburg CA", "Antioch CA", "Brentwood CA",
        "Oakley CA", "Livermore", "Pleasanton", "Dublin CA",
        "San Ramon", "Danville CA",
        # San Diego neighbourhoods
        "Downtown San Diego", "Gaslamp Quarter", "Mission Valley SD",
        "La Jolla", "Pacific Beach SD", "North Park SD", "Chula Vista",
        "Mission Hills SD", "Hillcrest", "University Heights SD",
        "Normal Heights", "Kensington SD", "Talmadge", "City Heights SD",
        "College Area SD", "Rolando", "El Cajon", "Santee CA",
        "Lakeside CA", "Ramona CA", "Spring Valley CA", "Lemon Grove",
        "National City CA", "Coronado", "Imperial Beach", "Bonita CA",
        # Sacramento neighbourhoods
        "Downtown Sacramento", "Midtown Sacramento", "Oak Park Sacramento",
        "Natomas", "Rancho Cordova", "Elk Grove", "Folsom CA",
        "Roseville CA", "Rocklin", "Lincoln CA", "Auburn CA",
        "Davis CA", "Woodland CA", "West Sacramento", "Citrus Heights",
        "Antelope CA", "North Highlands", "Rio Linda", "Elverta",
        # General California zone types
        "the school zone", "the hospital district", "the business district",
        "the residential zone", "the waterfront", "the transit hub",
        "the arts district", "the industrial zone", "the university area",
        "the shopping district", "the entertainment district",
        "the port area", "the airport corridor", "the stadium district",
        "the tech corridor", "the medical center area",
    ],
    "speed": ["10", "15", "20", "25", "30", "35", "40", "45", "50", "55"],
    "time": [
        "peak hours", "rush hour", "morning rush", "evening rush",
        "school hours", "school drop-off time", "school pickup time",
        "evenings", "weekends", "overnight", "daytime", "off-peak hours",
        "the morning commute", "the evening commute", "lunchtime",
        "late night", "early morning", "midday", "the afternoon",
        "before dawn", "post-rush hour", "the weekend morning",
        "Saturday morning", "Sunday evening", "Friday afternoon",
        "the holiday period", "game day traffic", "concert hours",
    ],
    "day": [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "Saturday", "Sunday", "weekdays", "weekends", "school days",
        "public holidays", "game days", "during the holidays",
        "on Labor Day", "on Thanksgiving", "on the Fourth of July",
        "on Memorial Day", "on New Year's Day", "on Christmas Day",
        "during the Rose Bowl", "during the LA Marathon",
        "during Fleet Week", "during Pride weekend",
        "during the California State Fair",
    ],
    "direction": [
        "northbound", "southbound", "eastbound", "westbound",
        "inbound", "outbound", "left", "right", "center",
        "HOV", "express", "local", "fast", "carpool",
    ],
    "intersection": [
        # Los Angeles
        "the Sunset and Vine intersection",
        "the Hollywood and Highland interchange",
        "the I-405 and I-10 interchange",
        "the US-101 and I-405 interchange",
        "the I-110 and I-105 interchange",
        "the I-5 and I-10 interchange",
        "the Wilshire and Western intersection",
        "the Vermont and Sunset crossing",
        "the Lincoln and Venice intersection",
        "the La Cienega and Wilshire junction",
        "the Sepulveda and Ventura intersection",
        "the Cahuenga and Hollywood junction",
        "the Figueroa and Adams crossing",
        "the Harbor and Century interchange",
        "the Crenshaw and Slauson junction",
        "the Western and Manchester crossing",
        "the Vermont and Manchester intersection",
        "the Normandie and Florence junction",
        "the Avalon and Imperial crossing",
        "the Central and Compton intersection",
        "the Long Beach and Imperial junction",
        "the Alameda and Rosecrans crossing",
        "the Pacific Coast Hwy and Lincoln junction",
        "the Lincoln and Pico intersection",
        "the 26th St and Santa Monica Blvd crossing",
        "the Bundy and Olympic intersection",
        "the Barrington and Wilshire junction",
        "the Centinela and Venice crossing",
        "the Overland and Venice intersection",
        "the Motor and National junction",
        "the Robertson and Venice crossing",
        "the La Brea and Sunset intersection",
        "the Fairfax and Melrose junction",
        "the Highland and Hollywood crossing",
        "the Vine and Fountain intersection",
        "the Cahuenga and Fountain junction",
        # Bay Area
        "the Market and Castro intersection",
        "the Van Ness and Geary junction",
        "the 19th Ave and Holloway crossing",
        "the Mission and 16th St intersection",
        "the Bay Bridge toll plaza",
        "the Golden Gate Bridge toll plaza",
        "the I-880 and I-80 interchange",
        "the US-101 and I-280 interchange",
        "the I-580 and I-680 interchange",
        "the I-80 and I-580 interchange",
        "the US-101 and SR-92 interchange",
        "the I-880 and SR-92 interchange",
        "the I-680 and SR-24 interchange",
        "the Market and Van Ness intersection",
        "the Embarcadero and Market crossing",
        "the Mission and Cesar Chavez junction",
        "the Valencia and 24th St intersection",
        "the Folsom and 6th St crossing",
        "the Howard and 4th St intersection",
        "the King and 3rd St junction",
        # San Diego
        "the I-5 and I-8 interchange",
        "the I-15 and I-163 interchange",
        "the Harbor and Broadway crossing SD",
        "the El Cajon and University junction",
        "the I-805 and SR-94 interchange",
        "the I-5 and SR-52 interchange",
        "the I-15 and SR-56 interchange",
        "the I-8 and I-805 interchange",
        "the Mission Valley and Hotel Circle junction",
        "the Sports Arena and Midway crossing",
        # Sacramento
        "the I-5 and I-80 interchange Sacramento",
        "the J St and 16th crossing Sacramento",
        "the Watt Ave and I-80 junction",
        "the Business 80 and US-50 interchange",
        "the I-5 and US-50 interchange Sacramento",
        "the Florin and Stockton Blvd crossing",
        "the Folsom and Bradshaw junction",
        "the Sunrise and US-50 interchange",
        "the Capital City Freeway and Watt Ave junction",
    ],
}

# ── Templates ─────────────────────────────────────────────────────────────────
# Fix — cross-label contamination:
# Each category now includes ~15 templates that contain surface keywords
# from OTHER categories. This prevents the model learning shortcuts like
# "mph → speed_limit_reduction" or "signal → signal_retiming".
# Examples: a congestion_charge template that mentions signals; a
# signal_retiming template that mentions mph; etc.
# The label is determined by the INTENT of the query, not its keywords.

TEMPLATES: dict[str, list[str]] = {
    "speed_limit_reduction": [
        # ── Core templates ────────────────────────────────────────────────────
        "What happens to traffic flow on {road} if we reduce the speed limit to {speed}mph?",
        "Simulate a {speed}mph speed limit on {road} during {time}",
        "Show me the pollution impact of lowering speeds near {area}",
        "What would a {speed}mph zone on {road} do to congestion during {time}?",
        "Model the effect of cutting vehicle speeds on {road} on {day}",
        "If we drop the speed limit to {speed}mph on {road}, what happens to emissions?",
        "Simulate reduced speed limits in {area} during {time} and show air quality impact",
        "What does traffic look like on {road} with a {speed}mph cap on {day}?",
        "Show me travel time changes if speeds on {road} are reduced to {speed}mph",
        "How does a {speed}mph limit near {area} affect pollution levels on {day}?",
        "Run a simulation of a {speed}mph speed zone on {road} during {time}",
        "What is the emissions impact of a {speed}mph limit along {road}?",
        "Model a {speed}mph corridor on {road} and show congestion effects on {day}",
        "How would lowering the speed limit to {speed}mph affect air quality near {area}?",
        "Simulate the effect of a speed reduction to {speed}mph on {road} on {day}",
        "What does a {speed}mph cap do to vehicle throughput on {road} during {time}?",
        "Show congestion changes if {road} is set to {speed}mph during {time}",
        "If speeds are capped at {speed}mph near {area}, what happens to pollution?",
        "Model the impact of reducing speed to {speed}mph near {area} on {day}",
        "What are the traffic effects of a {speed}mph zone on {road} during {time}?",
        "How does a {speed}mph speed limit on {road} change NOx levels on {day}?",
        "Simulate a school zone speed of {speed}mph near {area} and show safety impact",
        "What would a {speed}mph limit do to pedestrian safety near {area}?",
        "Show me PM2.5 levels near {area} if speeds drop to {speed}mph on {day}",
        "Model energy consumption change if {road} has a {speed}mph cap during {time}",
        "If vehicles slow to {speed}mph on {road}, how does CO2 output change?",
        "What is the throughput impact of a {speed}mph restriction on {road} on {day}?",
        "Simulate {speed}mph variable speed limits on {road} during {time}",
        "Show me how a {speed}mph zone near {area} affects emergency response times",
        "How does reducing speeds to {speed}mph on {road} affect freight delivery times?",
        "What happens to cyclist safety near {area} with a {speed}mph vehicle limit?",
        "Model the effect of a {speed}mph advisory speed on {road} during {time}",
        "Simulate lowering the speed limit to {speed}mph at {intersection} on {day}",
        "What does air quality look like near {area} if {road} is capped at {speed}mph?",
        "Show me the noise reduction impact of a {speed}mph limit on {road} on {day}",
        "How would a {speed}mph zone on {road} affect bus journey times during {time}?",
        "If we implement {speed}mph on {road}, what happens to rear-end collision risk?",
        "Simulate a reduced speed of {speed}mph on {road} and show energy savings",
        "What is the pollution reduction from a {speed}mph cap near {area} on {day}?",
        "Model a dynamic {speed}mph limit on {road} activated during {time}",
        # ── Cross-label templates (intent = speed limit; keywords from other labels) ──
        # mentions signals → model must NOT route to signal_retiming
        "If we lower the speed limit to {speed}mph on {road}, how do signal wait times change at {intersection}?",
        "Would a {speed}mph cap on {road} make signal coordination less necessary during {time}?",
        "Simulate {speed}mph on {road} and compare with signal retiming at {intersection} for pollution impact",
        "How does a {speed}mph zone interact with existing signal timing at {intersection} on {day}?",
        "If we cut speeds to {speed}mph near {area}, do we still need to retime signals on {road}?",
        # mentions lane → model must NOT route to lane_closure
        "Would a {speed}mph limit on {road} reduce the need to close lanes during {time}?",
        "Simulate a {speed}mph cap on the {direction} lane of {road} and show throughput on {day}",
        "If we reduce speed to {speed}mph on {road}, how does the {direction} lane perform during {time}?",
        "Model the safety benefit of {speed}mph on {road} compared to adding a lane near {area}",
        # mentions bus/transit → model must NOT route to public_transport_increase
        "How does a {speed}mph vehicle speed cap on {road} affect bus journey times during {time}?",
        "Would lowering speeds to {speed}mph near {area} improve bus reliability more than adding services?",
        "Simulate {speed}mph on {road} and show whether bus delays decrease on {day}",
        # mentions toll/charge → model must NOT route to congestion_charge
        "Is a {speed}mph limit on {road} more effective than a toll for reducing emissions near {area}?",
        "Model the pollution reduction from {speed}mph on {road} vs a congestion charge in {area} on {day}",
        "Would a {speed}mph cap on {road} during {time} achieve the same traffic reduction as a toll?",
    ],
    "lane_closure": [
        # ── Core templates ────────────────────────────────────────────────────
        "What happens to traffic if we close {road} on {day}?",
        "Simulate closing the {direction} lane on {road} during {time}",
        "Show me congestion impact of a lane closure on {road} on {day}",
        "What does a single-lane restriction on {road} do to travel times?",
        "Model the effect of removing the {direction} lane near {area}",
        "If {road} is reduced to one lane during {time}, what is the knock-on effect?",
        "Simulate a {direction} lane closure on {road} and show pollution change",
        "What would congestion look like near {area} if a lane is removed on {day}?",
        "Show me how traffic redistributes if {road} is closed on {day}",
        "How does a lane reduction on {road} during {time} affect surrounding roads?",
        "Run a simulation of closing the {direction} lane on {road} during {time}",
        "What is the spillover effect if {road} loses a lane near {area}?",
        "Model a full closure of {road} on {day} and show diversion routes",
        "How would removing the {direction} lane on {road} affect commute times?",
        "Simulate one lane blocked on {road} during {time} and show delay impact",
        "What does traffic look like at {intersection} if {road} is partially closed?",
        "Show the congestion impact of a {direction} lane removal near {area} on {day}",
        "If we shut the {direction} lane on {road}, how does it affect {area}?",
        "Model the pollution change from a lane closure on {road} during {time}",
        "What happens to vehicles at {intersection} if {road} is reduced to one lane?",
        "Simulate a rolling lane closure on {road} during {time} and show queue lengths",
        "What is the freight delay impact of closing {road} on {day}?",
        "Show me emergency vehicle access if the {direction} lane on {road} is closed",
        "Model the effect of a {direction} HOV lane conversion on {road} during {time}",
        "If {road} is closed near {area}, which alternate routes get congested?",
        "Simulate a lane closure at {intersection} during {time} and show signal impact",
        "What happens to bus routes if the {direction} lane on {road} is removed?",
        "Show me how cyclists and pedestrians are affected by a lane closure on {road}",
        "Model the air quality change from reduced vehicle flow on {road} on {day}",
        "What is the delay to transit if {road} loses its {direction} lane on {day}?",
        "Simulate overnight lane closure on {road} and show next-morning impact",
        "How does a temporary lane closure on {road} affect {area} businesses?",
        "What would closing {road} entirely on {day} do to the {area} road network?",
        "Show traffic queue length at {intersection} if {road} drops to one lane",
        "Model the carbon impact of lane closure congestion on {road} during {time}",
        "If the {direction} lane on {road} is converted to bike lane, what changes?",
        "Simulate weekend lane closure on {road} near {area} and show visitor impact",
        "What happens to pollution near {area} if {road} traffic is forced to reroute?",
        "Show me the delay impact of a {direction} lane closure at {intersection}",
        "Model diversion traffic through {area} if {road} is closed on {day}",
        # ── Cross-label templates (intent = lane closure; keywords from other labels) ──
        # mentions speed/mph → model must NOT route to speed_limit_reduction
        "If we close the {direction} lane on {road}, do vehicle speeds naturally drop below {speed}mph?",
        "Simulate closing {road} to one lane and show whether speeds fall to {speed}mph near {area}",
        "Would a lane removal on {road} during {time} reduce speeds enough to avoid a {speed}mph limit?",
        "Model speed impacts on {road} when the {direction} lane is closed on {day}",
        # mentions signal → model must NOT route to signal_retiming
        "How does a lane closure at {intersection} affect signal timing during {time}?",
        "If the {direction} lane on {road} is closed, do signals at {intersection} need retiming on {day}?",
        "Show queue buildup at {intersection} signals when {road} drops to one lane during {time}",
        "Simulate closing a lane on {road} and show whether signal phases at {intersection} need adjustment",
        # mentions bus/transit → model must NOT route to public_transport_increase
        "What happens to bus services on {road} if the {direction} lane is closed during {time}?",
        "Model bus delay on {road} from a {direction} lane closure near {area} on {day}",
        "If we close the {direction} lane on {road}, how badly are transit routes affected during {time}?",
        # mentions toll/charge → model must NOT route to congestion_charge
        "Would a lane closure on {road} during {time} have a bigger impact than a congestion charge near {area}?",
        "Simulate a {direction} lane closure on {road} and compare diversion routes to a tolled scenario on {day}",
        "Is closing the {direction} lane on {road} more disruptive than charging vehicles to enter {area}?",
    ],
    "signal_retiming": [
        # ── Core templates ────────────────────────────────────────────────────
        "What happens to traffic flow if we retime the signals on {road}?",
        "Simulate adjusted signal phases at {intersection} during {time}",
        "Show me the impact of optimizing traffic lights at {road} junction",
        "What does congestion look like at {intersection} with retimed signals on {day}?",
        "Model the effect of increasing green-light duration at {intersection} during {time}",
        "If we synchronize the lights along {road}, how does travel time change?",
        "Simulate adaptive signal control at {intersection} and show throughput impact",
        "What would shorter red-light cycles on {road} do to congestion during {time}?",
        "Show me pollution levels near {area} if signals at {intersection} are optimized",
        "How does coordinating traffic lights on {road} affect flow on {day}?",
        "Run a simulation of signal retiming on {road} during {time}",
        "What is the throughput gain from optimizing {intersection} signals on {day}?",
        "Model green wave progression along {road} and show travel time impact",
        "How would adaptive signals at {intersection} reduce idling and emissions?",
        "Simulate extended green phases on {road} during {time} and show queue length",
        "What does queue length look like at {intersection} with retimed signals?",
        "Show congestion reduction if signals on {road} are coordinated on {day}",
        "If we reduce cycle time at {intersection}, what happens to throughput?",
        "Model the emission impact of synchronizing lights along {road} during {time}",
        "What are the delay reductions from signal optimization near {area} on {day}?",
        "Simulate transit signal priority on {road} during {time} and show bus delays",
        "What happens to pedestrian wait times if signals at {intersection} are retimed?",
        "Show me the fuel savings from reduced idling if {road} lights are coordinated",
        "Model the impact of longer pedestrian crossing phases at {intersection} on {day}",
        "How does signal coordination on {road} affect emergency vehicle response?",
        "Simulate connected vehicle signal optimization near {area} during {time}",
        "What is the NOx reduction from less idling if {intersection} signals are retimed?",
        "Show traffic flow improvement on {road} with adaptive signal control on {day}",
        "Model the effect of removing a signal at {intersection} on {day}",
        "What happens to cyclist safety if signal timing at {intersection} is adjusted?",
        "Simulate pre-timed vs adaptive signals on {road} and compare congestion levels",
        "How would signal retiming on {road} affect school pickup traffic on {day}?",
        "Show me queue spillback reduction at {intersection} with optimized signals",
        "Model the pedestrian and vehicle conflict reduction at {intersection} on {day}",
        "What is the throughput difference between current and optimized signals on {road}?",
        "Simulate coordinated signals across {area} during {time} and show network impact",
        "How does a green wave on {road} during {time} affect downstream intersections?",
        "Show the energy savings from reduced stop-and-go traffic on {road} on {day}",
        "Model the effect of bus priority signals on {road} near {area} during {time}",
        "What happens to travel times through {area} if all signals on {road} are retimed?",
        # ── Cross-label templates (intent = signal retiming; keywords from other labels) ──
        # mentions mph/speed → model must NOT route to speed_limit_reduction
        "If we retime signals at {intersection}, do vehicles naturally slow below {speed}mph during {time}?",
        "Would optimizing lights on {road} reduce speeds enough to avoid a {speed}mph zone near {area}?",
        "Simulate signal coordination on {road} and show whether average speeds drop to {speed}mph on {day}",
        "Model the speed impact of retiming {intersection} signals vs enforcing a {speed}mph limit on {day}",
        "How does green wave timing on {road} interact with a {speed}mph advisory during {time}?",
        # mentions lane → model must NOT route to lane_closure
        "If we retime signals at {intersection}, can we avoid closing the {direction} lane on {road}?",
        "Simulate signal optimization on {road} and compare to a {direction} lane closure for throughput on {day}",
        "Would coordinating lights on {road} during {time} compensate for losing a lane near {area}?",
        "Model whether retiming {intersection} signals removes the need for a {direction} lane restriction",
        # mentions bus/transit → model must NOT route to public_transport_increase
        "Simulate transit signal priority on {road} and show how it compares to adding bus services on {day}",
        "Would bus priority signals on {road} during {time} be more effective than increasing bus frequency near {area}?",
        "Model the bus journey time improvement from signal retiming on {road} vs adding extra bus runs on {day}",
        # mentions toll/charge → model must NOT route to congestion_charge
        "Is retiming signals on {road} more cost-effective than a congestion charge for reducing delays near {area}?",
        "Simulate signal optimization at {intersection} and compare pollution reduction to a toll on {road} on {day}",
        "Would coordinating lights along {road} during {time} reduce congestion as much as charging vehicles to enter {area}?",
    ],
    "congestion_charge": [
        # ── Core templates ────────────────────────────────────────────────────
        "What happens to traffic volumes on {road} if a congestion charge is introduced?",
        "Simulate a congestion charge zone around {area} during {time}",
        "Show me pollution impact if vehicles are charged to enter {area} on {day}",
        "What does traffic look like on {road} with a toll applied during {time}?",
        "Model the effect of a road pricing scheme in {area} on vehicle numbers",
        "If a congestion fee is applied to {road} during {time}, how does air quality change?",
        "Simulate a low-emission charging zone in {area} and show pollution reduction",
        "What would a pay-per-use scheme on {road} do to congestion on {day}?",
        "Show me how a charging zone around {area} affects traffic distribution",
        "How does introducing a toll on {road} during {time} impact surrounding roads?",
        "Run a simulation of congestion pricing on {road} during {time}",
        "What is the traffic reduction if vehicles pay to enter {area} on {day}?",
        "Model the modal shift effect of a congestion charge near {area}",
        "How would a peak-hour toll on {road} change commuter behavior?",
        "Simulate dynamic pricing on {road} during {time} and show emission changes",
        "What does vehicle volume look like on {road} if a charge is introduced on {day}?",
        "Show air quality improvement from a congestion charge zone in {area}",
        "If a toll is added to {road} during {time}, how does traffic redistribute?",
        "Model the pollution reduction from charging vehicles entering {area} on {day}",
        "What are the traffic effects of a congestion zone covering {area} during {time}?",
        "Simulate a variable congestion charge on {road} and show peak hour response",
        "What is the PM2.5 reduction from a low-traffic zone in {area} on {day}?",
        "Show me how freight traffic responds to a toll on {road} during {time}",
        "Model the equity impact of congestion pricing near {area} on {day}",
        "How does a mileage-based user fee on {road} affect travel patterns?",
        "Simulate cordon pricing around {area} and show border road congestion",
        "What happens to parking demand near {area} if a congestion charge is added?",
        "Show the transit ridership increase from congestion pricing on {road} on {day}",
        "Model the revenue generated by tolling {road} during {time}",
        "What is the carbon reduction from a congestion charge zone in {area}?",
        "Simulate a zero-emission zone in {area} and show NOx impact on {day}",
        "How does congestion pricing on {road} affect delivery costs for {area} businesses?",
        "Show me diversion traffic on parallel roads if {road} is tolled during {time}",
        "Model the health impact of reduced vehicle emissions in {area} after pricing",
        "What happens to school run traffic near {area} if a charge is introduced on {day}?",
        "Simulate a weekend congestion charge on {road} and show leisure trip impact",
        "How does dynamic tolling at {intersection} affect throughput during {time}?",
        "Show the traffic reduction on {road} from a congestion charge starting on {day}",
        "Model the combined effect of congestion pricing and transit improvement in {area}",
        "What is the spillover congestion on nearby roads if {road} is tolled on {day}?",
        # ── Cross-label templates (intent = congestion charge; keywords from other labels) ──
        # mentions mph/speed → model must NOT route to speed_limit_reduction
        "Would a congestion charge on {road} during {time} be more effective than lowering speeds to {speed}mph near {area}?",
        "Simulate tolling {road} and show whether average vehicle speeds drop to {speed}mph as a side effect on {day}",
        "Model the emission reduction from charging vehicles on {road} vs a {speed}mph speed cap near {area}",
        "If we introduce a fee to enter {area}, do we also need a {speed}mph limit on {road} during {time}?",
        # mentions signal → model must NOT route to signal_retiming
        "How does a congestion charge zone in {area} affect signal demand at {intersection} during {time}?",
        "If we toll {road} during {time}, do signal timings at {intersection} need adjustment on {day}?",
        "Simulate a charging zone in {area} and show the effect on signal queue lengths at {intersection}",
        "Would a cordon charge around {area} reduce traffic enough that signals at {intersection} can be simplified?",
        # mentions lane → model must NOT route to lane_closure
        "Is tolling {road} during {time} more effective than closing the {direction} lane for reducing congestion near {area}?",
        "Simulate a congestion fee on {road} and compare to a {direction} lane closure for emissions on {day}",
        "Would a charge to enter {area} reduce demand enough to avoid closing lanes on {road} during {time}?",
        # mentions bus/transit → model must NOT route to public_transport_increase
        "Simulate congestion pricing on {road} combined with increased bus frequency and show which reduces car volumes more near {area}",
        "Would a toll on {road} during {time} drive modal shift to transit near {area} without adding bus services?",
        "Model the transit ridership gain from a congestion charge in {area} vs adding bus routes on {road} on {day}",
    ],
    "public_transport_increase": [
        # ── Core templates ────────────────────────────────────────────────────
        "What happens to congestion on {road} if we increase bus frequency?",
        "Simulate more frequent bus services along {road} during {time}",
        "Show me pollution impact of increasing public transport near {area} on {day}",
        "What does traffic look like on {road} if extra buses are added during {time}?",
        "Model the effect of boosting bus frequency in {area} on private vehicle use",
        "If train frequency through {area} is increased, how does congestion change?",
        "Simulate improved public transport near {road} and show emissions reduction",
        "What would adding a new bus route through {area} do to traffic on {day}?",
        "Show me travel time changes if public transport on {road} is strengthened",
        "How does increasing bus services in {area} during {time} affect pollution levels?",
        "Run a simulation of doubling bus frequency on {road} during {time}",
        "What is the car reduction effect of more transit options near {area}?",
        "Model the emissions impact of adding a new bus line through {area} on {day}",
        "How would increased metro frequency near {area} reduce road congestion?",
        "Simulate bus rapid transit on {road} and show throughput changes",
        "What does modal shift look like if bus frequency doubles near {area}?",
        "Show congestion reduction on {road} if transit capacity is increased on {day}",
        "If a new bus route is added along {road}, what happens to car volumes?",
        "Model the pollution change from increased public transport in {area} during {time}",
        "What are the traffic effects of improved transit frequency near {area} on {day}?",
        "Simulate adding light rail through {area} and show long-term congestion impact",
        "What is the NOx reduction from modal shift to transit near {area} on {day}?",
        "Show me how additional bus stops on {road} affect traffic flow during {time}",
        "Model the effect of free transit days on vehicle volumes near {area} on {day}",
        "How would a new express bus service on {road} reduce commute times?",
        "Simulate 24-hour bus service near {area} and show overnight congestion change",
        "What happens to parking demand near {area} if transit frequency doubles?",
        "Show the carbon savings from modal shift if buses are added on {road} on {day}",
        "Model the effect of electric bus fleet on emissions near {area} during {time}",
        "What is the congestion reduction on {road} from adding a transit-only lane?",
        "Simulate increased Metrolink frequency through {area} on {day}",
        "How does a bus-on-shoulder program on {road} reduce delays during {time}?",
        "Show me the health impact of reduced car use near {area} from better transit",
        "Model the effect of real-time bus information on ridership near {area}",
        "What happens to school traffic near {area} if school bus frequency increases?",
        "Simulate ferry service expansion near {area} and show road traffic reduction",
        "How does bike-transit integration near {area} affect car volumes on {day}?",
        "Show the congestion impact on {road} from adding a park-and-ride near {area}",
        "Model the emission reduction from electric bus deployment on {road} during {time}",
        "What is the travel time saving from bus rapid transit on {road} on {day}?",
        # ── Cross-label templates (intent = transit increase; keywords from other labels) ──
        # mentions mph/speed → model must NOT route to speed_limit_reduction
        "If we double bus frequency on {road}, do average car speeds rise above {speed}mph during {time}?",
        "Would adding transit near {area} reduce road congestion enough that a {speed}mph limit becomes unnecessary?",
        "Simulate increased bus services on {road} and show whether car speeds stabilise around {speed}mph on {day}",
        "Model how adding transit capacity in {area} affects the need for a {speed}mph speed restriction on {road}",
        # mentions signal → model must NOT route to signal_retiming
        "If we increase bus frequency on {road}, how do signal wait times change at {intersection} during {time}?",
        "Would bus priority signals at {intersection} be enough, or do we need more buses on {road} on {day}?",
        "Simulate adding bus services to {road} and show whether signal coordination at {intersection} also improves",
        "Model the interaction between higher bus frequency on {road} and signal timing at {intersection} on {day}",
        # mentions lane → model must NOT route to lane_closure
        "If we add a transit-only lane on {road}, how does that differ from simply increasing bus frequency near {area}?",
        "Would more buses on {road} during {time} reduce car volumes enough to avoid closing the {direction} lane?",
        "Simulate increasing public transport near {area} and compare with a {direction} lane closure for congestion on {day}",
        # mentions toll/charge → model must NOT route to congestion_charge
        "Would increasing bus frequency near {area} achieve the same modal shift as a congestion charge on {road} on {day}?",
        "Simulate doubling transit frequency in {area} and compare car volume reduction to a toll on {road} during {time}",
        "Model whether adding bus services on {road} is more equitable than charging vehicles to enter {area} on {day}",
    ],
}

# ── Hard examples ─────────────────────────────────────────────────────────────
# Manually written examples that are genuinely ambiguous at the keyword level.
# Each would fool a keyword-matching classifier but has a clear correct label
# based on the primary intervention being simulated.
# ~40 per label, 200 total. No slots — appear exactly as written.
# These are mixed into each label's pool during generation so the model
# must learn intent rather than surface keyword cues.

HARD_EXAMPLES: dict[str, list[str]] = {
    "speed_limit_reduction": [
        "What is the fastest way to reduce pollution on Wilshire Blvd without closing lanes or adding tolls?",
        "We already have frequent buses on the 101 freeway — what else can we do to cut emissions near Hollywood?",
        "Would cutting vehicle speeds near the school zone be more effective than adding more school buses?",
        "Our signal timing is already optimised on Ventura Blvd — what other lever can we pull to reduce accidents?",
        "If we can't close lanes on Interstate 405, what is the next best option to improve safety near the LAX corridor?",
        "The congestion charge proposal was rejected — what vehicle-level intervention can we model instead?",
        "We want to reduce rear-end collisions on the 101 freeway without restricting lanes or adding tolls",
        "Is there anything we can do on Sunset Blvd to cut noise pollution without bus expansion or road closures?",
        "The city wants lower emissions on Highland Ave but refuses to close lanes — what can we model?",
        "Without pricing vehicles or closing roads, what is the most direct way to reduce CO2 on Pacific Coast Highway?",
        "We need to improve pedestrian safety near the university area without changing bus routes or signal cycles",
        "The freight lobby blocked the lane closure proposal — what alternative reduces truck emissions on Alameda St?",
        "We already retimed all the signals on Vermont Ave — what is left to reduce congestion at peak hours?",
        "Is there a way to reduce stopping distances on the 405 freeway without touching signal timing or transit?",
        "The council wants quieter streets near Nob Hill without bus changes or road pricing — what do we simulate?",
        "How can we reduce idling near the hospital district without adding buses or retiming signals?",
        "We need a quick win on emissions for the Downtown LA air quality report — no new transit, no tolls",
        "Without any infrastructure spend, what is the most effective way to cut vehicle emissions on {road}?",
        "Can we reduce CO2 on {road} during {time} without closing lanes, adding transit, or introducing charges?",
        "The environmental team wants pollution reduced on {road} — no lane changes, no tolls, no new bus routes",
        "We modelled signal retiming and transit expansion near {area} — what is the third option for {day}?",
        "After the congestion charge failed politically, what is the quickest operational change on {road}?",
        "If lane closures are off the table and the bus network is already expanded, what reduces speeds near {area}?",
        "The safety audit recommends reducing vehicle kinetic energy near {intersection} — what intervention does that?",
        "How do we reduce PM2.5 on {road} during {time} without any infrastructure changes?",
        "We need to reduce the 85th percentile speed on {road} near {area} — what policy achieves that directly?",
        "The community near {area} wants calmer traffic on {road} but opposes road closures and new bus stops",
        "Our budget only allows a paint-and-sign intervention on {road} — what reduces speeds near {area}?",
        "Is there a way to reduce vehicle throughput on {road} during {time} without closing lanes or adding tolls?",
        "We want drivers to slow down approaching {intersection} on {day} — no signals involved, no lane changes",
        "What is the lowest-cost intervention to reduce emissions on {road} near {area} during {time}?",
        "The transit operator says they cannot add more buses on {road} — what else reduces congestion near {area}?",
        "We need to reduce fuel consumption on {road} during {time} using only on-road regulation",
        "Without changing the number of lanes or the frequency of buses, how do we reduce emissions near {area}?",
        "The road pricing scheme on {road} was abandoned — what direct vehicle behaviour change can we model?",
        "How do we reduce the severity of accidents on {road} near {area} using only a regulatory change?",
        "We want to reduce CO2 from {road} on {day} — signals are fine, buses are fine, lanes are staying open",
        "The school near {area} needs calmer traffic on {road} during {time} — no lane closures, no new services",
        "What is the single most direct way to reduce kinetic energy of vehicles on {road} near {area}?",
        "We need to reduce stopping risk on {road} near {area} during {time} — no pricing, no transit changes",
    ],
    "lane_closure": [
        "We want to reduce throughput on Wilshire Blvd without raising prices or changing signal cycles",
        "The speed limit on Interstate 405 is already as low as we can go — what is the next step for congestion?",
        "We have retimed all signals on Ventura Blvd — the congestion is still there — what do we try next?",
        "Bus frequency on Hollywood Blvd has been doubled — traffic is still bad — what do we model next?",
        "Without changing the speed limit or the fare structure, how do we reduce vehicle flow on Sunset Blvd?",
        "The pricing scheme on the 110 freeway was rejected — what capacity intervention can we model instead?",
        "We want to convert vehicle space to cycling infrastructure on Cahuenga Blvd — what does that do to traffic?",
        "We need to reduce vehicle throughput on {road} near {area} without any pricing or speed change",
        "The transit authority cannot run more buses on {road} — what else reduces vehicle demand near {area}?",
        "Without a toll and without lowering speed limits, how do we reduce the number of vehicles on {road}?",
        "Emergency maintenance is required on {road} — model the traffic impact before we start work on {day}",
        "We want to reallocate road space on {road} to pedestrians — show the vehicle impact during {time}",
        "The cycle lane proposal for {road} near {area} would remove one vehicle lane — what are the traffic effects?",
        "We need to simulate a utility works closure on {road} at {intersection} for planning purposes on {day}",
        "If we remove a lane from {road} to add a dedicated bus lane, what happens to car journey times during {time}?",
        "The road resurfacing on {road} will restrict to one lane — model the knock-on effects near {area} on {day}",
        "We want to test what happens if {road} loses capacity near {area} without any price signal involved",
        "A water main burst has closed {road} near {intersection} — what is the diversion route impact on {day}?",
        "We are proposing a plaza at {intersection} which removes traffic lanes — model the congestion effect",
        "How does the {area} road network cope if {road} is taken out of service for infrastructure works on {day}?",
        "We want to dedicate the {direction} lane on {road} to freight-only during {time} — show the car impact",
        "Without speed reductions or transit changes, what happens to journey times if {road} loses a lane near {area}?",
        "We need to assess whether {area} businesses can still be served if {road} is narrowed on {day}",
        "The city wants to test road diet interventions on {road} — model the effect during {time}",
        "If the {direction} lane on {road} is converted to a protected bike lane, how does traffic redistribute near {area}?",
        "We are planning a public event in {area} that will require partial closure of {road} on {day} — model impacts",
        "Show me what happens to the {area} network if construction blocks {road} for three months starting {day}",
        "The council wants to pedestrianise part of {road} near {area} — model the traffic diversion on {day}",
        "We need to close {road} overnight near {intersection} for bridge maintenance — show next-morning impact",
        "What is the best diversion route if we close {road} near {area} during {time}?",
        "Model the emergency response impact if {road} near {area} is reduced to one lane on {day}",
        "We want to ban through-traffic on {road} near {area} — what does that do to parallel roads during {time}?",
        "A protest march will block {road} at {intersection} on {day} — show the traffic redistribution",
        "The city is trialling a car-free zone near {area} — model the effects on {road} during {time}",
        "We need to assess the traffic impact of a new building site that will block one lane of {road} for six months",
        "The port expansion near {area} will close {road} during {time} — model the freight diversion",
        "We want to run a pop-up market on {road} near {area} on {day} — what road capacity is lost?",
        "Model the traffic impact of removing the slip road from {intersection} permanently",
        "We are testing whether {road} can operate safely with one lane closed during {time} near {area}",
        "Show the network effect of closing {road} between {intersection} and {area} for the whole of {day}",
    ],
    "signal_retiming": [
        "We want to reduce idling on Wilshire Blvd without closing lanes, adding buses, or changing speed limits",
        "The congestion charge on Ventura Blvd was rejected — what software-only change can we model?",
        "Bus frequency on Vermont Ave is already at maximum — what else reduces stop-and-go congestion near Koreatown?",
        "We cannot close lanes on the 101 freeway — is there a control-room intervention that reduces queue length?",
        "Without any physical infrastructure change, how do we reduce fuel burn on Hollywood Blvd during morning rush?",
        "We want to improve pedestrian crossing times at the Wilshire and Western intersection without removing lanes",
        "The speed limit on Sunset Blvd is already as low as the code allows — what else reduces emissions?",
        "We need a zero-capital intervention to reduce congestion on {road} during {time}",
        "Without changing speed limits or transit frequency, how do we reduce queue length at {intersection} on {day}?",
        "We want to prioritise emergency vehicles on {road} near {area} using existing infrastructure only",
        "The budget for {road} is zero — show me what operational change reduces congestion most during {time}",
        "We want to reduce pedestrian fatalities at {intersection} without building new infrastructure on {day}",
        "How can we reduce idling emissions at {intersection} using only the existing traffic controller hardware?",
        "The transit priority on {road} is too slow — can we improve bus journey times without adding new services?",
        "We want to reduce stop-and-go patterns on {road} near {area} — no lane changes, no pricing, no speed limit",
        "Is there a way to improve throughput at {intersection} on {day} without any physical road works?",
        "We need to reduce the cycle time at {intersection} to help emergency services near {area} during {time}",
        "The council approved only a software upgrade for {road} — what traffic outcome can we model?",
        "We want to cut vehicle idling near {area} during {time} using only traffic management tools",
        "Without closing lanes or adding tolls, how do we reduce queue spillback at {intersection} on {day}?",
        "Model the throughput gain from optimising green phases at {intersection} without any physical change on {day}",
        "We want to test connected vehicle signal optimization on {road} — no lane changes, no speed enforcement",
        "How much can we reduce CO2 on {road} near {area} using only signal phase adjustments during {time}?",
        "The transport authority wants a quick win on journey times for {road} on {day} — signals only",
        "We need to improve freight throughput on {road} during {time} using existing intersection hardware near {area}",
        "Without a bus lane or new vehicles, how do we improve bus reliability on {road} near {area}?",
        "We want to reduce pedestrian wait times at {intersection} on {day} without building new crossings",
        "Model the effect of giving cyclists a head-start phase at {intersection} during {time} near {area}",
        "We want to reduce rear-end collisions on {road} at {intersection} using only timing changes on {day}",
        "The school near {area} wants safer crossing times at {intersection} on school days — no new infrastructure",
        "How do we reduce vehicle conflict at {intersection} during {time} without narrowing lanes or adding police?",
        "We need to improve green wave progression along {road} on {day} using only controller adjustments",
        "Model the NOx reduction from reducing unnecessary red-light stops on {road} near {area} during {time}",
        "We want to test adaptive signal control on {road} at {intersection} before committing to hardware purchase",
        "The air quality team needs emission reductions at {intersection} on {day} — infrastructure budget is zero",
        "Without changing the road layout near {area}, how do we improve throughput on {road} during {time}?",
        "Model the fuel savings from reducing stop cycles at {intersection} on {road} near {area} on {day}",
        "We want to improve the pedestrian crossing phase at {intersection} without changing vehicle lane allocation",
        "How much journey time can we save on {road} during {time} with signal coordination alone near {area}?",
        "The road geometry on {road} cannot be changed — show the best signal-only intervention at {intersection}",
    ],
    "congestion_charge": [
        "We want to reduce vehicle volumes on Wilshire Blvd without changing road layout or signal cycles",
        "We have already lowered the speed limit on the 101 freeway — traffic is still heavy — what is the next lever?",
        "Signal retiming on Ventura Blvd is complete — we still need to reduce car numbers near Hollywood",
        "Bus frequency near Downtown LA has been doubled — private car use is unchanged — what do we try next?",
        "We want to use price signals to manage demand on Sunset Blvd without any infrastructure change",
        "The road diet on {road} has been implemented — we now need to manage the remaining vehicle demand near {area}",
        "We want to reduce vehicle kilometres travelled in {area} on {day} using a demand-side instrument",
        "Without any road works or transit expansion, how do we reduce car trips to {area} during {time}?",
        "We need to fund road maintenance near {area} — what pricing mechanism can we model on {road}?",
        "The city wants to reduce car dependency near {area} using economics rather than physical restrictions on {day}",
        "Model the behavioural response if we introduce a financial disincentive to drive on {road} during {time}",
        "We have already reduced speed limits and retimed signals on {road} — what demand-side tool is next?",
        "Without closing lanes or adding buses, how do we shift commuters away from {road} during {time}?",
        "We want to test whether {area} residents are price-sensitive to driving on {road} on {day}",
        "The environmental team wants to use market mechanisms to reduce emissions near {area} during {time}",
        "We need to model user charging on {road} to fund the transit expansion proposed for {area}",
        "Without any speed enforcement or signal changes, how do we reduce peak demand on {road} near {area}?",
        "We want to shift freight from {road} to rail near {area} using a financial instrument on {day}",
        "Model the revenue potential of a user charge on {road} near {area} to fund public transport on {day}",
        "We need to reduce discretionary trips through {area} on {road} during {time} — no lane changes available",
        "The bus network near {area} cannot be expanded this year — what price-based intervention reduces car use?",
        "Signal changes on {road} are already maxed out — we need a demand management tool for {area} on {day}",
        "We want to test cordon pricing around {area} before presenting it to the council — model the traffic effect",
        "Without physical road changes, how do we reduce vehicle flow on {road} near {area} during {time}?",
        "We need to reduce peak hour demand on {road} using a fiscal instrument rather than a capacity change",
        "Model the equity implications of charging vehicles to enter {area} compared to reducing bus fares on {day}",
        "We want to use dynamic pricing on {road} to smooth demand curves near {area} during {time}",
        "The speed limit on {road} is already low and lanes are full — what reduces vehicle numbers near {area}?",
        "We need to generate revenue from {road} near {area} to fund infrastructure maintenance on {day}",
        "Without adding lanes or buses, how do we make {road} less attractive to through-traffic near {area}?",
        "We want to test whether a financial signal reduces school-run traffic near {area} on {road} during {time}",
        "Model the traffic redistribution if we introduce user charges on {road} while keeping signal timing unchanged",
        "We want to reduce car use near {area} on {day} using incentives and disincentives rather than road changes",
        "The cycling infrastructure near {area} is in place — now we need to make driving on {road} less attractive",
        "We need a policy instrument that reduces car volumes on {road} without restricting physical capacity near {area}",
        "How do we use pricing to shift demand from {road} to {area} transit options during {time} on {day}?",
        "Model the impact of variable user fees on {road} to flatten the demand curve near {area} during {time}",
        "We want to reduce the number of single-occupancy vehicles entering {area} on {day} — pricing only",
        "Without a bus lane or speed restriction, what is the most effective tool to reduce car volumes near {area}?",
        "Model the effect of a financial deterrent on {road} during {time} on vehicle numbers near {area} on {day}",
    ],
    "public_transport_increase": [
        "We want to reduce car use near Downtown LA without changing road layout, speed limits, or signal cycles",
        "The congestion charge on Wilshire Blvd failed politically — what supply-side intervention reduces car trips?",
        "We have already lowered speeds and retimed signals on the 101 freeway — car volumes are still too high",
        "Lane closures on Ventura Blvd have been ruled out — how else do we reduce private vehicle use near Hollywood?",
        "Without any road pricing or physical changes to {road}, how do we reduce car dependency near {area}?",
        "We want to improve access to {area} on {day} for residents without cars — no new roads, no tolls",
        "The road network near {area} is at capacity — what intervention reduces demand from the supply side?",
        "We need to reduce car volumes near {area} during {time} by making the alternative to driving more attractive",
        "Without restricting vehicles or charging them, how do we reduce private car trips to {area} on {day}?",
        "We want to shift commuters from {road} to another mode during {time} — no pricing, no road changes",
        "The council wants to reduce emissions near {area} on {day} by investing in alternatives to the car",
        "We need to improve mobility near {area} for people who cannot drive — what do we simulate on {day}?",
        "Without closing lanes or adding tolls, how do we make {road} less congested by reducing car demand near {area}?",
        "We want to reduce parking demand near {area} on {day} by making it easier not to drive",
        "Model the car volume reduction on {road} during {time} if we invest in the alternative to driving near {area}",
        "The cycling network near {area} is complete — what is the next mode-shift intervention on {road}?",
        "We want to reduce school-run traffic near {area} by improving the alternative to the car on {day}",
        "Without changing signal cycles or speed limits, how do we reduce vehicle kilometres in {area} on {day}?",
        "We need to improve the door-to-door journey time from {area} to {road} for non-drivers during {time}",
        "The city wants to reduce car dependency near {area} by investing in transport alternatives on {day}",
        "We want to model the car volume effect of making {area} easier to reach without a car during {time}",
        "Without any pricing mechanism, how do we shift {road} commuters to a lower-emission mode near {area}?",
        "We need to reduce single-occupancy vehicle trips to {area} on {day} — no road restrictions involved",
        "Model the congestion reduction on {road} if residents near {area} had a faster non-car option during {time}",
        "We want to serve {area} workers who currently have no good alternative to driving on {road} on {day}",
        "Without changing the price of driving, how do we reduce car volumes on {road} near {area} during {time}?",
        "The air quality near {area} needs to improve on {day} — model an intervention that removes cars by attraction",
        "We want to reduce traffic on {road} by giving people near {area} a reason not to use their cars during {time}",
        "Model the emission benefit of reducing car trips to {area} by improving non-car journey times on {day}",
        "We need to serve the growing population near {area} without increasing car traffic on {road} during {time}",
        "The speed limit is already low and signals are retimed on {road} — what demand-side pull reduces car use?",
        "We want to attract drivers away from {road} near {area} during {time} without any financial penalty",
        "Model what happens to car volumes on {road} if travel time by the alternative mode is halved near {area}",
        "We need to improve transport equity near {area} on {day} for residents without access to a vehicle",
        "Without road pricing or physical changes, how do we reduce peak-hour car volumes near {area} on {road}?",
        "We want to test whether improving non-car options near {area} reduces vehicle demand on {road} during {time}",
        "Model the mode shift from {road} if we make the journey to {area} by non-car means as fast as by car",
        "We need to cut car use near {area} on {day} using a positive incentive rather than a restriction",
        "Without changing the road or its price, how do we reduce traffic on {road} near {area} during {time}?",
        "The community near {area} wants less traffic on {road} on {day} — model an approach that does not penalise drivers",
    ],
}


# Fix: removed ALL CAPS transform — bert-base-uncased lowercases all input,
# so UPPER == lower after tokenization, producing silent near-duplicates.
# Replaced with semantically neutral rewrites that preserve label intent
# while adding surface diversity BERT can actually distinguish.

_QUERY_PREFIXES = [
    "Can you show me ",
    "I need to understand ",
    "Please model ",
    "Help me simulate ",
    "I want to see ",
    "Can you model ",
    "Run a simulation — ",
    "For the digital twin: ",
    "Quick question: ",
    "Operator query: ",
]

_FILLER_PHRASES = [
    " as part of our urban planning review",
    " for the next city council meeting",
    " to support the environmental impact assessment",
    " for our traffic engineering team",
    " as a sensitivity test",
    " to compare against baseline",
    " before we submit the proposal",
    " using the latest sensor data",
    " across the full road network",
    " including downstream effects",
]

def apply_surface_variation(text: str, rng: random.Random) -> str:
    """
    Apply lightweight surface variation that survives bert-base-uncased tokenization.
    Three transforms, each applied independently with low probability:
      1. Prepend a neutral operator-style prefix  (~12% of examples)
      2. Append a neutral context filler phrase    (~10% of examples)
      3. Lowercase the whole string               (~15% of examples)
         — safe because bert-base-uncased does this anyway; keeps casing
           consistent rather than randomly mixing cases.
    ALL CAPS removed: bert-base-uncased lowercases input, so UPPER and lower
    produce identical token sequences — pure duplicate inflation.
    """
    # Prefix
    if rng.random() < 0.12:
        prefix = rng.choice(_QUERY_PREFIXES)
        # lower-case first char of original text after a prefix
        text = prefix + text[0].lower() + text[1:]

    # Suffix filler
    if rng.random() < 0.10:
        # strip trailing punctuation before appending
        text = text.rstrip("?.") + rng.choice(_FILLER_PHRASES)

    # Lowercase (consistent with tokenizer behaviour, adds minor variety)
    if rng.random() < 0.15:
        text = text.lower()

    return text


# ── Core generation ───────────────────────────────────────────────────────────

def fill_template(template: str, rng: random.Random) -> str:
    """Replace all {slot} placeholders with a random value from SLOTS."""
    placeholders = re.findall(r"\{(\w+)\}", template)
    result = template
    for ph in placeholders:
        value = rng.choice(SLOTS[ph])
        result = result.replace(f"{{{ph}}}", value, 1)
    return result


def generate_examples(
    templates: dict[str, list[str]],
    hard_examples: dict[str, list[str]],
    n_per_category: int,
    rng: random.Random,
) -> list[dict]:
    """
    For each category, generate n_per_category examples.

    Strategy:
      1. FORCE all HARD_EXAMPLES into the selected set first (guaranteed inclusion).
         Hard examples are filled (slots resolved) and reserved before any
         template sampling. This guarantees the model sees examples that require
         semantic understanding rather than keyword matching.
      2. Fill remaining slots from Cartesian product of (template x slot_combo).
         Security fix #8: templates whose Cartesian product exceeds
         MAX_CARTESIAN_PER_TEMPLATE are skipped for exhaustive expansion
         and sampled randomly instead, preventing RAM exhaustion.
      3. Shuffle the final selected set so hard examples are not bunched at start.
      4. Security fix #9: exact duplicate sentences are tracked and reported.
    """
    records: list[dict] = []

    for label, tmpl_list in templates.items():

        # ── Step 1: force hard examples into selected (guaranteed) ────────────
        forced: list[str] = []
        for hard_tmpl in hard_examples.get(label, []):
            forced.append(fill_template(hard_tmpl, rng))
        n_hard = len(forced)

        # Remaining slots to fill from template pool
        n_from_pool = n_per_category - n_hard

        # ── Step 2: build template pool for remaining examples ────────────────
        pool: list[str] = []
        for tmpl in tmpl_list:
            phs = re.findall(r"\{(\w+)\}", tmpl)
            if not phs:
                pool.append(tmpl)
                continue

            slot_options = [SLOTS[ph] for ph in phs]

            # Security fix #8: guard against RAM exhaustion from huge
            # Cartesian products on multi-slot templates
            product_size = 1
            for opts in slot_options:
                product_size *= len(opts)

            if product_size > MAX_CARTESIAN_PER_TEMPLATE:
                for _ in range(min(product_size, MAX_CARTESIAN_PER_TEMPLATE)):
                    pool.append(fill_template(tmpl, rng))
            else:
                for combo in cartesian_product(*slot_options):
                    text = tmpl
                    for ph, val in zip(phs, combo):
                        text = text.replace(f"{{{ph}}}", val, 1)
                    pool.append(text)

        rng.shuffle(pool)

        if len(pool) >= n_from_pool:
            from_pool = pool[:n_from_pool]
        else:
            from_pool = pool[:]
            while len(from_pool) < n_from_pool:
                from_pool.append(fill_template(rng.choice(tmpl_list), rng))

        # ── Step 3: combine, shuffle, apply surface variation ─────────────────
        selected = forced + from_pool
        rng.shuffle(selected)

        # Security fix #9: detect and report exact duplicates
        unique_selected = list(dict.fromkeys(selected))
        dupe_count = len(selected) - len(unique_selected)
        if dupe_count > 0:
            print(
                f"  ⚠  {label}: {dupe_count} duplicate sentences detected "
                f"({len(unique_selected)} unique out of {len(selected)})",
                file=sys.stderr,
            )

        for text in selected:
            text = apply_surface_variation(text, rng)
            records.append({"text": text, "label": label})

        print(f"  {label:<30} {len(selected):>6} examples  "
              f"(pool: {len(pool):,}  hard: {n_hard}  dupes: {dupe_count})")

    return records


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BERT training data.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write policy_labels.jsonl into (must be within the project root).",
    )
    args = parser.parse_args()

    output_path = _resolve_output_path(args.output_dir)

    # Security fix #4: refuse to generate a runaway dataset
    if EXAMPLES_PER_CATEGORY > MAX_EXAMPLES_PER_CATEGORY:
        raise ValueError(
            f"EXAMPLES_PER_CATEGORY ({EXAMPLES_PER_CATEGORY}) exceeds the hard "
            f"ceiling of {MAX_EXAMPLES_PER_CATEGORY}. Refusing to continue."
        )

    # Security fix #7: deduplicate slot values before generation
    cleaned_slots = _deduplicate_slots(SLOTS)

    # Security fix #6: validate all template slot references exist in SLOTS
    _validate_slots(TEMPLATES, cleaned_slots)
    # Also validate hard example slots
    _validate_slots(HARD_EXAMPLES, cleaned_slots)
    print("✓  Slot validation passed — all template and hard-example references are valid\n")

    # Patch SLOTS in place with deduplicated values for generation
    SLOTS.update(cleaned_slots)

    rng = random.Random(SEED)

    total_expected = EXAMPLES_PER_CATEGORY * len(TEMPLATES)
    print(f"Generating training examples…")
    print(f"  {EXAMPLES_PER_CATEGORY:,} per category x "
          f"{len(TEMPLATES)} categories = {total_expected:,} total\n")

    records = generate_examples(TEMPLATES, HARD_EXAMPLES, EXAMPLES_PER_CATEGORY, rng)
    rng.shuffle(records)

    # Security fix #3: atomic write
    os.makedirs(output_path.parent, exist_ok=True)
    tmp_path = Path(str(output_path) + ".tmp")  # fix #10: explicit suffix, not with_suffix()
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    # ── Validation summary ────────────────────────────────────────────────────
    counts = Counter(r["label"] for r in records)
    total  = sum(counts.values())

    print(f"\n✓  Written {total:,} records to {output_path}")
    print("\nLabel distribution:")
    for label, count in sorted(counts.items()):
        bar = "█" * (count // 500)
        print(f"  {label:<30}  {count:>6}  {bar}")

    # Security fix #5: explicit raises instead of assert
    min_total = len(TEMPLATES) * EXAMPLES_PER_CATEGORY
    if total < min_total:
        raise ValueError(f"Expected ≥{min_total:,} records, got {total:,}.")
    under = {lbl: c for lbl, c in counts.items() if c < EXAMPLES_PER_CATEGORY}
    if under:
        raise ValueError(
            f"Some categories have fewer than {EXAMPLES_PER_CATEGORY:,} examples: {under}"
        )
    print("\n✓  All acceptance criteria met.")


if __name__ == "__main__":
    main()