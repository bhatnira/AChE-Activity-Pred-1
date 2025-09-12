#!/usr/bin/env python3

"""
Demonstration of the difference between log10 and natural log (ln) conversions for IC50 values
"""

import numpy as np

def compare_conversions():
    print("=== Comparison of log10 vs natural log (ln) conversions ===\n")
    
    # Test with some example predicted values
    test_values = [4.0, 5.0, 6.0, 7.0, 8.0]
    
    print("Predicted Value | log10 → IC50 (nM) | ln → IC50 (nM)")
    print("-" * 55)
    
    for val in test_values:
        # Log10 conversion (used by RDKit, Circular, Graph models)
        ic50_log10 = 10 ** val
        
        # Natural log conversion (used by ChemBERTa)
        ic50_ln = np.exp(val)
        
        print(f"{val:13.1f} | {ic50_log10:13.1f} | {ic50_ln:11.1f}")
    
    print("\nKey Differences:")
    print("• RDKit/Circular/Graph models: IC50 = 10^(predicted_value)")
    print("• ChemBERTa model: IC50 = exp(predicted_value) = e^(predicted_value)")
    print("\nFor the same predicted value:")
    print("• log10 conversion gives much larger IC50 values")
    print("• ln conversion gives more reasonable IC50 values in the nM range")
    
    print("\nExample: If model predicts 7.0")
    print(f"• log10: IC50 = 10^7 = {10**7:,.0f} nM = {10**7/1000:,.0f} μM")
    print(f"• ln: IC50 = e^7 = {np.exp(7):.1f} nM = {np.exp(7)/1000:.3f} μM")

if __name__ == "__main__":
    compare_conversions()
