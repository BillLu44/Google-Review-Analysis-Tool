#!/usr/bin/env python3
# test_pipeline.py
# Author: Bill Lu
# Description: Run a suite of sample reviews through the full NLP pipeline for validation and timing.

import sys
import os
import time
import json
import uuid

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pipeline.logger import get_logger
from pipeline.preprocessing import preprocess_text
from pipeline.rule_based import rule_based_sentiment
from pipeline.overall_sentiment import analyze_sentiment
from pipeline.absa import analyze_absa
from pipeline.emotion import detect_emotion
from pipeline.sarcasm import detect_sarcasm
from pipeline.fusion import fuse_signals
from utils.output_formatter import save_results_to_file

# Initialize logger
logger = get_logger(__name__)

# Diverse sample reviews for testing - realistic Google reviews from various businesses
SAMPLE_REVIEWS = [
    # Furniture Store Reviews
    {
        'review_id': str(uuid.uuid4()),
        'text': "Bought a sectional here last month and it's already sagging. The sales guy promised it would last years but the cushions are completely flat. Quality is terrible for the price we paid."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "IKEA West Sacramento has everything you need for home furnishing! Staff was super helpful when I couldn't find the hex keys. Assembly instructions could be clearer but overall great value."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Delivery was supposed to be between 10-2pm. They showed up at 6pm without calling. Then they scratched my hardwood floors bringing in the dining table. Manager was apologetic but damage is done."
    },
    
    # Auto Service Reviews  
    {
        'review_id': str(uuid.uuid4()),
        'text': "Quick oil change turned into a 3-hour nightmare. They said I needed $800 in repairs but I got a second opinion - nothing was wrong. Avoid this place, they're trying to scam you."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Honest mechanics are hard to find but Tony's Auto is the real deal. Fixed my transmission for half what the dealer quoted. Been going here for 5 years, never had an issue."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Great service, terrible location. Parking is impossible and they're always backed up. Make an appointment or you'll wait forever. Work quality is good though."
    },
    
    # Hair Salon Reviews
    {
        'review_id': str(uuid.uuid4()),
        'text': "Asked for subtle highlights and walked out looking like a zebra. Stylist didn't listen to what I wanted and now I have to pay someone else to fix this disaster. So disappointed."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Maria is an absolute artist! She transformed my damaged hair into something gorgeous. The salon is clean, staff is friendly, and prices are reasonable. Will definitely be back!"
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Booked online for 2pm, didn't get seen until 3:30pm. No apology, no explanation. The cut was fine but the disrespect for my time was unacceptable."
    },
    
    # Medical/Dental Reviews
    {
        'review_id': str(uuid.uuid4()),
        'text': "Dr. Johnson's office is always running late but he's thorough and really cares about his patients. The hygienist was gentle during my cleaning. Billing department needs work though."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Worst dental experience of my life. Filling fell out after two days and they wanted to charge me again to fix their mistake. Rude receptionist, overpriced, and incompetent work."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Emergency root canal on a Saturday - Dr. Kim saved the day! Pain relief was immediate and the follow-up care was excellent. Staff went above and beyond."
    },
    
    # Tech/Phone Store Reviews
    {
        'review_id': str(uuid.uuid4()),
        'text': "Bought a 'new' phone that was clearly refurbished. Screen had scratches and battery life is awful. When I complained, they said all sales are final. Shady business practices."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Best Buy Geek Squad fixed my laptop faster than expected and for less than quoted. The technician explained everything clearly and even showed me how to prevent future issues."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Sales person was pushy and tried to sell me insurance I didn't need. Prices are higher than online but sometimes you need it today. Staff knowledge varies widely."
    },
    
    # Gym/Fitness Reviews
    {
        'review_id': str(uuid.uuid4()),
        'text': "Signed up for a 'no commitment' membership and they've been charging me for 6 months after I canceled. Equipment is old and half of it doesn't work. Save your money."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Planet Fitness gets a bad rap but this location is clean, well-maintained, and the staff is friendly. Not fancy but perfect for basic workouts. Can't beat the price."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Love the group fitness classes here! Instructors are motivating and the variety keeps things interesting. Locker rooms could use an update but overall great value."
    },
    
    # Pet Store/Vet Reviews
    {
        'review_id': str(uuid.uuid4()),
        'text': "Brought my sick cat here as an emergency. Dr. Martinez was compassionate and honest about treatment options. Expensive but they saved my buddy's life. Forever grateful."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Petco grooming butchered my dog's coat. Asked for a trim and they practically shaved him bald. Poor guy is embarrassed and cold. Manager offered discount but damage is done."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Staff really knows their stuff about aquarium setup. Helped me pick the right fish and equipment for a beginner. Fish are healthy and beautiful. Great customer service."
    },
    
    # Home Services Reviews
    {
        'review_id': str(uuid.uuid4()),
        'text': "Plumber showed up drunk, made a mess, and somehow made my leak worse. Had to call someone else to fix his 'repair'. Absolutely unprofessional. Avoid at all costs."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Same-day AC repair in 100-degree weather - these guys are lifesavers! Fair pricing, clean work, and they explained everything. Will use them for all HVAC needs."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Landscaping crew did beautiful work on our backyard transformation. Project took longer than expected due to weather but the results exceeded our expectations. Worth every penny."
    },
    
    # Shopping/Retail Reviews
    {
        'review_id': str(uuid.uuid4()),
        'text': "Target's grocery section is hit or miss. Produce looks tired and prices are higher than regular grocery stores. Convenient for one-stop shopping but quality suffers."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Nordstrom customer service is unmatched. Returned a dress I wore once (with tags) no questions asked. Sales associates are knowledgeable and not pushy. You pay more but get more."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Walmart self-checkout is a joke. Half the scanners don't work and you need assistance for everything. Employees act like helping customers is a burden. Shop elsewhere if you can."
    },
    
    # Entertainment/Movie Theater Reviews
    {
        'review_id': str(uuid.uuid4()),
        'text': "AMC theaters are overpriced for what you get. Sticky floors, broken seats, and $15 popcorn. The IMAX experience was good but not worth the premium. Netflix at home is better."
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Local drive-in theater is a hidden gem! Nostalgic experience with current movies. Bring blankets and snacks. Sound quality through car radio is surprisingly good. Fun date night!"
    },
    {
        'review_id': str(uuid.uuid4()),
        'text': "Regal Cinemas remodel looks great but they somehow made the seats smaller. Concession lines are always understaffed. Movie selection is standard but facilities are clean."
    }
]


def run_tests():
    """
    Processes each sample review through all pipeline stages and logs results and timings.
    """
    total_start = time.time()
    all_results = []

    for sample in SAMPLE_REVIEWS:
        rid = sample['review_id']
        text = sample['text']
        logger.info(f"\n=== Processing review_id={rid} ===")

        # 1. Preprocessing
        t0 = time.time()
        pre = preprocess_text(text)
        t1 = time.time()

        # 2. Rule-based
        rule_output = rule_based_sentiment(text)
        t2 = time.time()

        # 3. Transformer sentiment
        sentiment_output = analyze_sentiment(text)
        t3 = time.time()

        # 4. Aspect-based sentiment
        absa_output = analyze_absa(text)
        t4 = time.time()

        # 5. Emotion detection
        emotion_output = detect_emotion(text)
        t5 = time.time()

        # 6. Sarcasm detection
        sarcasm_output = detect_sarcasm(text)
        t6 = time.time()

        # Adapt sentiment_output for fusion model's expected sentiment_score (-1 to 1 float)
        transformer_sentiment_score_for_fusion = 0.0
        if sentiment_output['sentiment'] == 1: # positive
            transformer_sentiment_score_for_fusion = sentiment_output['confidence_score']
        elif sentiment_output['sentiment'] == -1: # negative
            transformer_sentiment_score_for_fusion = -sentiment_output['confidence_score']

        # 7. Fusion
        signals = {
            'rule_score': rule_output['rule_score'],
            'rule_polarity': rule_output['rule_polarity'],
            'sentiment_score': transformer_sentiment_score_for_fusion,
            'sentiment_confidence': sentiment_output['confidence_score'],
            'num_pos_aspects': absa_output['num_pos_aspects'],
            'num_neg_aspects': absa_output['num_neg_aspects'],
            'avg_aspect_score': absa_output['avg_aspect_score'],
            'avg_aspect_confidence': absa_output['avg_aspect_confidence'],
            'emotion_score': emotion_output['emotion_score'],
            'emotion_confidence': emotion_output['emotion_confidence'],
            'emotion_distribution': emotion_output['emotion_distribution'],
            'sarcasm_score': sarcasm_output['sarcasm_score'],
            'sarcasm_confidence': sarcasm_output['sarcasm_confidence']
        }
        fused = fuse_signals(signals)
        t7 = time.time()

        # Compile results with timing
        timing_data = {
            'preprocessing_time': round(t1 - t0, 3),
            'rule_based_time': round(t2 - t1, 3),
            'transformer_sentiment_time': round(t3 - t2, 3),
            'absa_time': round(t4 - t3, 3),
            'emotion_time': round(t5 - t4, 3),
            'sarcasm_time': round(t6 - t5, 3),
            'fusion_time': round(t7 - t6, 3),
            'total_time': round(t7 - t0, 3)
        }

        result = {
            'review_id': rid,
            'text': text,
            'timing': timing_data,  # Add timing data
            'signals': signals,
            'fused': fused,
            'aspects': absa_output['aspect_details'],
            'emotion_scores': emotion_output,
            'sarcasm': sarcasm_output,
            'rule': rule_output,
            'sentiment': sentiment_output
        }

        all_results.append(result)

        # Log structured result (for debugging)
        logger.info(json.dumps(result, indent=2))

    total_time = time.time() - total_start
    logger.info(f"\nProcessed {len(SAMPLE_REVIEWS)} reviews in {total_time:.2f}s")
    
    # Save human-readable results to output file
    output_file = save_results_to_file(all_results)
    logger.info(f"📄 Human-readable results saved to: {output_file}")
    print(f"\n✅ Pipeline completed! Results saved to: {output_file}")


if __name__ == '__main__':
    run_tests()
