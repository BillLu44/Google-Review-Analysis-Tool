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

def format_aspect_table(aspects: List[Dict]) -> str:
    """Format aspects into a readable table with context phrases"""
    if not aspects:
        return "No aspects detected"
    
    lines = []
    lines.append("┌─────────────────┬───────────┬───────────┬─────────────────────────────────┐")
    lines.append("│     Aspect      │ Sentiment │   Score   │           Context Phrase        │")
    lines.append("├─────────────────┼───────────┼───────────┼─────────────────────────────────┤")
    
    for aspect in aspects:
        name = aspect.get('aspect', 'unknown')[:15].ljust(15)
        sentiment = aspect.get('sentiment_label', 'neutral')
        score = aspect.get('sentiment_score', 0.0)
        context = aspect.get('context_phrase', 'N/A')[:31]  # Truncate long phrases
        emoji = format_sentiment_emoji(sentiment)
        
        # If context is too long, truncate with ellipsis
        if len(aspect.get('context_phrase', '')) > 31:
            context = context[:28] + "..."
        context = context.ljust(31)
        
        lines.append(f"│ {name} │ {sentiment[:9].ljust(9)} │ {score:+6.2f} {emoji} │ {context} │")
    
    lines.append("└─────────────────┴───────────┴───────────┴─────────────────────────────────┘")
    return "\n".join(lines)

def format_emotion_scores(emotion_data: Dict) -> str:
    """Format emotion scores into readable text"""
    if not emotion_data or not isinstance(emotion_data, dict):
        return "😶 Emotion Analysis:\n   No emotion data available"
    
    lines = []
    lines.append("😊 Emotion Analysis:")
    
    emotions = emotion_data.get('all_emotion_scores', {})
    if emotions and isinstance(emotions, dict):
        # Sort emotions by score
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        for emotion, score in sorted_emotions[:5]:  # Show top 5 emotions
            bar = format_confidence_bar(score, width=15)
            lines.append(f"   {emotion.capitalize():12} {bar}")
    else:
        lines.append("   No emotion scores available")
    
    return "\n".join(lines)

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
    
    # Top aspects mentioned with sample phrases
    all_aspects = {}
    aspect_phrases = {}  # Store sample phrases for each aspect
    for result in results:
        for aspect in result.get('aspects', []):
            aspect_name = aspect.get('aspect', 'unknown')
            if aspect_name not in all_aspects:
                all_aspects[aspect_name] = {'positive': 0, 'negative': 0, 'neutral': 0}
                aspect_phrases[aspect_name] = []
            sentiment = aspect.get('sentiment_label', 'neutral')
            all_aspects[aspect_name][sentiment] += 1
            
            # Store sample context phrase
            context = aspect.get('context_phrase', '')
            if context and len(aspect_phrases[aspect_name]) < 2:  # Store max 2 examples
                aspect_phrases[aspect_name].append(context[:50])  # Truncate long phrases
    
    if all_aspects:
        lines.append("🔍 TOP ASPECTS MENTIONED")
        sorted_aspects = sorted(all_aspects.items(), key=lambda x: sum(x[1].values()), reverse=True)[:5]
        for aspect, counts in sorted_aspects:
            total = sum(counts.values())
            pos_pct = counts['positive'] / total * 100
            neg_pct = counts['negative'] / total * 100
            lines.append(f"   {aspect.title():12} ({total:2d}x) - Pos: {pos_pct:4.1f}% | Neg: {neg_pct:4.1f}%")
            
            # Add sample phrases
            if aspect in aspect_phrases and aspect_phrases[aspect]:
                sample_phrases = aspect_phrases[aspect][:2]  # Show up to 2 examples
                for phrase in sample_phrases:
                    lines.append(f"     └─ \"{phrase}\"")
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
        emotion_data = result.get('emotion_scores', {})
        if emotion_data and isinstance(emotion_data, dict):
            emotions = emotion_data.get('all_emotion_scores', {})
            if emotions and isinstance(emotions, dict):
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

def format_signals_summary(signals: Dict) -> str:
    """Format model signals into readable text"""
    if not signals or not isinstance(signals, dict):
        return "No signal data available"
    
    lines = []
    lines.append("Model Signal Analysis:")
    
    # Process each signal
    for model_name, signal in signals.items():
        if isinstance(signal, dict):
            label = signal.get('label', 'unknown')
            confidence = signal.get('confidence', 0.0)
            lines.append(f"   {model_name}: {label.capitalize()} (confidence: {confidence:.3f})")
    
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
    lines.append("=" * 90)  # Made wider to accommodate context phrases
    lines.append(f"REVIEW #{review_id} ANALYSIS REPORT")
    lines.append("=" * 90)
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
    
    # Aspects with context
    aspects = result.get('aspects', [])
    lines.append("🔍 Aspect-Based Analysis:")
    lines.append(format_aspect_table(aspects))
    
    # Add detailed aspect context if available
    if aspects:
        lines.append("")
        lines.append("📖 Detailed Aspect Context:")
        for i, aspect in enumerate(aspects, 1):
            aspect_name = aspect.get('aspect', 'unknown')
            sentiment = aspect.get('sentiment_label', 'neutral')
            score = aspect.get('sentiment_score', 0.0)
            context = aspect.get('context_phrase', 'No context available')
            confidence = aspect.get('confidence', 0.0)
            
            emoji = format_sentiment_emoji(sentiment)
            lines.append(f"   {i}. {aspect_name.upper()} {emoji}")
            lines.append(f"      Sentiment: {sentiment} (score: {score:+.3f}, confidence: {confidence:.3f})")
            lines.append(f"      Context: \"{context}\"")
            lines.append("")
    
    lines.append("")
    
    # Emotions
    emotion_data = result.get('emotion_scores', {})
    lines.append(format_emotion_scores(emotion_data))
    lines.append("")
    
    # Sarcasm
    sarcasm = result.get('sarcasm', {})
    sarcasm_label = sarcasm.get('sarcasm_label', 'not_sarcastic')
    sarcasm_score = sarcasm.get('sarcasm_score', 0.0)
    sarcasm_confidence_val = sarcasm.get('sarcasm_confidence', 0.0)

    lines.append("🎭 Sarcasm Analysis:")
    lines.append(f"   Detection: {sarcasm_label}")
    lines.append(f"   Confidence: {sarcasm_confidence_val:.1%}")
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

def format_pipeline_timing(result: Dict) -> str:
    """Format pipeline timing information"""
    timing = result.get('timing', {})
    if not timing or not isinstance(timing, dict):
        return "Pipeline Timing: No timing data available"
    
    lines = []
    lines.append("⏱ Pipeline Timing:")
    
    # Process timing data
    total_time = timing.get('total_time', 0)
    lines.append(f"   Total processing time: {total_time:.3f}s")
    
    # Add individual component timings if available
    for component, time in timing.items():
        if component != 'total_time':
            lines.append(f"   {component}: {time:.3f}s")
    
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