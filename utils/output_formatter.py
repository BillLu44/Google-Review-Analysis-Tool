# utils/output_formatter.py
# Author: Bill Lu
# Description: Human-readable output formatting for pipeline results

import os
import json
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

def format_confidence_bar(confidence: float, width: int = 20) -> str:
    """Create a visual confidence bar"""
    filled = int(confidence * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {confidence:.1%}"

def format_sentiment_emoji(label: str) -> str:
    """Add emoji based on sentiment"""
    emoji_map = {
        'positive': '😊',
        'negative': '😞',
        'neutral': '😐'
    }
    return emoji_map.get(label, '❓')

def format_executive_summary(results: List[Dict]) -> str:
    """Create executive summary with key statistics"""
    if not results:
        return "No results to summarize"
    
    lines = []
    lines.append("📊 EXECUTIVE SUMMARY")
    lines.append("=" * 50)
    lines.append("")
    
    # Basic stats
    total_reviews = len(results)
    lines.append(f"📈 OVERVIEW")
    lines.append(f"   Total Reviews Analyzed: {total_reviews}")
    
    # Confidence stats
    confidences = [r.get('fused', {}).get('fused_confidence', 0) for r in results]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    high_conf = sum(1 for c in confidences if c >= 0.8)
    low_conf = sum(1 for c in confidences if c < 0.6)
    
    lines.append(f"   Average Confidence: {avg_confidence:.1%}")
    lines.append(f"   High Confidence (≥80%): {high_conf} ({high_conf/total_reviews*100:.1f}%)")
    lines.append(f"   Low Confidence (<60%): {low_conf} ({low_conf/total_reviews*100:.1f}%)")
    lines.append("")
    
    # Sentiment distribution
    sentiments = [r.get('fused', {}).get('fused_label', 'unknown') for r in results]
    sentiment_counts = {}
    for s in sentiments:
        sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
    
    lines.append("🎯 SENTIMENT DISTRIBUTION")
    for sentiment, count in sorted(sentiment_counts.items()):
        percentage = count / total_reviews * 100
        bar_length = int(percentage / 5)  # Scale bar to fit
        bar = "█" * bar_length
        emoji = format_sentiment_emoji(sentiment)
        lines.append(f"   {sentiment.capitalize():10} {count:3d} ({percentage:5.1f}%) {bar} {emoji}")
    lines.append("")
    
    # Top aspects mentioned
    all_aspects = {}
    for result in results:
        for aspect in result.get('aspects', []):
            aspect_name = aspect.get('aspect', 'unknown')
            if aspect_name not in all_aspects:
                all_aspects[aspect_name] = {'positive': 0, 'negative': 0, 'neutral': 0}
            sentiment = aspect.get('sentiment_label', 'neutral')
            all_aspects[aspect_name][sentiment] += 1
    
    if all_aspects:
        lines.append("🔍 TOP ASPECTS MENTIONED")
        sorted_aspects = sorted(all_aspects.items(), key=lambda x: sum(x[1].values()), reverse=True)[:5]
        for aspect, counts in sorted_aspects:
            total = sum(counts.values())
            pos_pct = counts['positive'] / total * 100
            neg_pct = counts['negative'] / total * 100
            lines.append(f"   {aspect.title():12} ({total:2d}x) - Pos: {pos_pct:4.1f}% | Neg: {neg_pct:4.1f}%")
        lines.append("")
    
    # Model performance insights
    lines.append("🤖 MODEL INSIGHTS")

    # Sarcasm detection stats — count only label=='sarcastic'
    sarcastic_count = sum(1 for r in results if r.get('sarcasm', {}).get('sarcasm_label') == 'sarcastic')
    if sarcastic_count > 0:
        lines.append(f"   Sarcasm Detected: {sarcastic_count} reviews ({sarcastic_count/total_reviews*100:.1f}%)")
    
    # Emotion distribution
    all_emotions = {}
    for result in results:
        emotions = result.get('emotion_scores', {})
        if emotions:
            top_emotion = max(emotions.keys(), key=lambda k: emotions[k])
            all_emotions[top_emotion] = all_emotions.get(top_emotion, 0) + 1
    
    if all_emotions:
        lines.append("   Top Emotions:")
        sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        for emotion, count in sorted_emotions:
            lines.append(f"     {emotion.title()}: {count} ({count/total_reviews*100:.1f}%)")
    
    lines.append("")
    lines.append("=" * 50)
    lines.append("")
    
    return "\n".join(lines)

def format_aspect_table(aspects: List[Dict]) -> str:
    """Format aspects into a readable table"""
    if not aspects:
        return "No aspects detected"
    
    lines = []
    lines.append("┌─────────────────┬───────────┬───────────┐")
    lines.append("│     Aspect      │ Sentiment │   Score   │")
    lines.append("├─────────────────┼───────────┼───────────┤")
    
    for aspect in aspects:
        name = aspect.get('aspect', 'unknown')[:15].ljust(15)
        sentiment = aspect.get('sentiment_label', 'neutral')
        score = aspect.get('sentiment_score', 0.0)
        emoji = format_sentiment_emoji(sentiment)
        
        lines.append(f"│ {name} │ {sentiment[:9].ljust(9)} │ {score:+6.2f} {emoji} │")
    
    lines.append("└─────────────────┴───────────┴───────────┘")
    return "\n".join(lines)

def format_emotion_scores(emotions: Dict[str, float]) -> str:
    """Format emotion scores with bars"""
    if not emotions:
        return "No emotions detected"
    
    # Sort by score descending
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    
    lines = []
    lines.append("Emotion Analysis:")
    for emotion, score in sorted_emotions:
        bar = format_confidence_bar(score, width=15)
        lines.append(f"  {emotion.capitalize():10} {bar}")
    
    return "\n".join(lines)

def format_pipeline_timing(result: Dict) -> str:
    """Format timing information"""
    timing_fields = [
        ('preprocessing_time', 'Preprocessing'),
        ('rule_based_time', 'Rule-based'),
        ('transformer_sentiment_time', 'Transformer'),
        ('absa_time', 'ABSA'),
        ('emotion_time', 'Emotion'),
        ('sarcasm_time', 'Sarcasm'),
        ('fusion_time', 'Fusion')
    ]
    
    lines = []
    lines.append("Performance Timing:")
    total_time = sum(result.get(field, 0) for field, _ in timing_fields)
    
    for field, label in timing_fields:
        time_ms = result.get(field, 0) * 1000  # Convert to ms
        lines.append(f"  {label:15} {time_ms:6.1f}ms")
    
    lines.append(f"  {'Total':15} {total_time * 1000:6.1f}ms")
    return "\n".join(lines)

def format_signals_summary(signals: Dict) -> str:
    """Format the fusion signals"""
    lines = []
    lines.append("Signal Analysis:")
    
    signal_info = [
        ('rule_score', 'Rule-based Score', lambda x: f"{x:+6.2f}"),
        ('sentiment_score', 'Transformer Score', lambda x: f"{x:+6.2f}"),
        ('num_pos_aspects', 'Positive Aspects', lambda x: f"{x:3.0f}"),
        ('num_neg_aspects', 'Negative Aspects', lambda x: f"{x:3.0f}"),
        ('avg_aspect_score', 'Avg Aspect Score', lambda x: f"{x:+6.2f}"),
        ('emotion_score', 'Emotion Score', lambda x: f"{x:6.2f}"),
        ('sarcasm_score', 'Sarcasm Score', lambda x: f"{x:6.2f}")
    ]
    
    for key, label, formatter in signal_info:
        value = signals.get(key, 0.0)
        formatted_value = formatter(value)
        lines.append(f"  {label:18} {formatted_value}")
    
    return "\n".join(lines)

def format_single_result(result: Dict) -> str:
    """Format a single review result into human-readable text"""
    review_id = result.get('review_id', 'Unknown')
    text = result.get('text', 'No text available')
    fused = result.get('fused', {})
    fused_label = fused.get('fused_label', 'unknown')
    fused_confidence = fused.get('fused_confidence', 0.0)
    
    # Header
    lines = []
    lines.append("=" * 70)
    lines.append(f"REVIEW #{review_id} ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Show original text first for easy reference
    lines.append("📝 ORIGINAL REVIEW TEXT")
    lines.append("-" * 30)
    # Wrap text if it's very long
    if len(text) > 200:
        lines.append(f'"{text[:200]}..."')
        lines.append(f"[Full text: {len(text)} characters]")
    else:
        lines.append(f'"{text}"')
    lines.append("")
    
    # Final Result (most important)
    emoji = format_sentiment_emoji(fused_label)
    confidence_bar = format_confidence_bar(fused_confidence)
    lines.append("🎯 FINAL SENTIMENT PREDICTION")
    lines.append(f"   Result: {fused_label.upper()} {emoji}")
    lines.append(f"   Confidence: {confidence_bar}")
    
    # Apply confidence threshold
    if fused_confidence < 0.6:
        lines.append("   ⚠️  LOW CONFIDENCE - Consider human review")
    elif fused_confidence >= 0.9:
        lines.append("   ✅ HIGH CONFIDENCE")
    
    lines.append("")
    
    # Detailed Analysis
    lines.append("📊 DETAILED ANALYSIS")
    lines.append("-" * 30)
    lines.append("")
    
    # Aspects
    aspects = result.get('aspects', [])
    lines.append("🔍 Aspect-Based Analysis:")
    lines.append(format_aspect_table(aspects))
    lines.append("")
    
    # Emotions
    emotions = result.get('emotion_scores', {})
    lines.append(format_emotion_scores(emotions))
    lines.append("")
    
    # Sarcasm
    sarcasm = result.get('sarcasm', {})
    sarcasm_label = sarcasm.get('sarcasm_label', 'not_sarcastic')
    sarcasm_score = sarcasm.get('sarcasm_score', 0.0)

    lines.append("🎭 Sarcasm Analysis:")
    lines.append(f"   Detection: {sarcasm_label}")
    lines.append(f"   Confidence: {sarcasm_score:.1%}")
    if sarcasm_label == 'sarcastic':
        lines.append("   ⚠️  Sarcasm detected - review context carefully")
    lines.append("")
    
    # Technical Details
    lines.append("⚙️  TECHNICAL DETAILS")
    lines.append("-" * 30)
    lines.append("")
    
    signals = result.get('signals', {})
    lines.append(format_signals_summary(signals))
    lines.append("")
    
    lines.append(format_pipeline_timing(result))
    lines.append("")
    
    return "\n".join(lines)

def save_results_to_file(results: List[Dict], filename: str = None) -> str:
    """Save formatted results to output file"""
    # Create output directory
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pipeline_results_{timestamp}.txt"
    
    filepath = output_dir / filename
    
    # Format all results
    lines = []
    lines.append("🤖 AI SENTIMENT ANALYSIS PIPELINE RESULTS")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Executive Summary FIRST
    lines.append(format_executive_summary(results))
    
    # Individual detailed results
    lines.append("📋 DETAILED INDIVIDUAL ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    
    for i, result in enumerate(results, 1):
        lines.append(format_single_result(result))
        if i < len(results):  # Add separator between results
            lines.append("\n\n")
    
    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    return str(filepath)