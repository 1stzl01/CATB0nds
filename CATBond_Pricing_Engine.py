import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    """Model type enumeration."""
    EVENT_BASED = "event_based"
    AGGREGATE = "aggregate"

@dataclass
class CATBondLayer:
    """CAT bond layer definition."""
    attachment: float
    limit: float
    name: str = ""
    
    @property
    def exhaustion(self) -> float:
        return self.attachment + self.limit

@dataclass
class PricingResult:
    """Pricing result container."""
    expected_loss: float
    gross_premium: float
    fair_spread: float
    risk_metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]

class EnhancedCATBondPricer:
    """
    Enhanced CAT Bond pricing engine supporting both event-based 
    and aggregate loss modeling approaches.
    """
    
    def __init__(self, model_type: ModelType = ModelType.EVENT_BASED, 
                 random_seed: int = 42):
        self.model_type = model_type
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def generate_event_losses(self, 
                            exposures: pd.DataFrame,
                            num_events: int = 10000,
                            frequency_lambda: float = 2.5) -> np.ndarray:
        """Generate event-based losses with improved statistical foundation."""
        
        # Create more sophisticated event catalog
        losses = []
        
        for event_id in range(num_events):
            # Simulate event characteristics
            event_magnitude = np.random.beta(2, 5) * 10  # Intensity scale 0-10
            event_footprint_size = int(np.random.exponential(5) + 3)  # Variable footprint
            
            total_loss = 0.0
            
            # Random geographic center for event
            center_lat = np.random.uniform(25.0, 27.0)
            center_lon = np.random.uniform(-82.0, -80.0)
            
            for _, exposure in exposures.iterrows():
                # Distance-based intensity decay
                distance = self._calculate_distance(
                    center_lat, center_lon, 
                    exposure.get('lat', center_lat), 
                    exposure.get('lon', center_lon)
                )
                
                # Intensity decreases with distance
                local_intensity = event_magnitude * np.exp(-distance / 50)  # 50km decay
                
                if local_intensity > 1.0:  # Minimum threshold for damage
                    # Construction-specific vulnerability
                    construction_type = exposure.get('construction_type', 'wood_frame')
                    vuln_curve = self._get_vulnerability_curve(construction_type)
                    
                    intensity_bin = min(10, max(1, int(local_intensity)))
                    damage_ratio = vuln_curve.get(intensity_bin, 0)
                    
                    loss = exposure['tiv'] * damage_ratio
                    total_loss += loss
            
            losses.append(total_loss)
        
        return np.array(losses)
    
    def generate_aggregate_losses(self, 
                                num_simulations: int = 10000,
                                loss_params: Optional[Dict] = None) -> np.ndarray:
        """Generate aggregate losses using calibrated distributions."""
        
        if loss_params is None:
            # Default parameters calibrated to historical hurricane data
            loss_params = {
                'distribution': 'lognormal',
                'mean_log': np.log(25) - 0.5 * (0.8)**2,  # ~$25M mean
                'sigma_log': 0.8,
                'max_loss': 1000  # Cap at $1B
            }
        
        if loss_params['distribution'] == 'lognormal':
            losses = np.random.lognormal(
                loss_params['mean_log'], 
                loss_params['sigma_log'], 
                num_simulations
            )
        elif loss_params['distribution'] == 'pareto':
            # Heavy-tailed distribution for extreme events
            losses = np.random.pareto(loss_params.get('alpha', 1.5), num_simulations)
            losses = losses * loss_params.get('scale', 10)
        else:
            raise ValueError(f"Unsupported distribution: {loss_params['distribution']}")
        
        return np.clip(losses, 0, loss_params.get('max_loss', 1000))
    
    def _calculate_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Calculate approximate distance in km between two points."""
        # Simplified distance calculation
        return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111  # ~111 km per degree
    
    def _get_vulnerability_curve(self, construction_type: str) -> Dict[int, float]:
        """Get construction-specific vulnerability curve."""
        curves = {
            'concrete': {i: min(1.0, 0.05 * i**1.8) for i in range(1, 11)},
            'masonry': {i: min(1.0, 0.08 * i**1.6) for i in range(1, 11)},
            'wood_frame': {i: min(1.0, 0.12 * i**1.4) for i in range(1, 11)},
            'mobile_home': {i: min(1.0, 0.20 * i**1.2) for i in range(1, 11)},
            'steel': {i: min(1.0, 0.03 * i**2.0) for i in range(1, 11)}
        }
        return curves.get(construction_type, curves['wood_frame'])
    
    def calculate_layer_payout(self, losses: np.ndarray, 
                             layer: CATBondLayer) -> np.ndarray:
        """Calculate layer payouts for given losses."""
        return np.maximum(
            np.minimum(losses - layer.attachment, layer.limit), 
            0
        )
    
    def calculate_comprehensive_metrics(self, 
                                      payouts: np.ndarray,
                                      layer: CATBondLayer) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        return {
            'expected_loss': np.mean(payouts),
            'std_dev': np.std(payouts),
            'max_payout': np.max(payouts),
            'var_95': np.percentile(payouts, 95),
            'var_99': np.percentile(payouts, 99),
            'var_99_9': np.percentile(payouts, 99.9),
            'tvar_95': np.mean(payouts[payouts >= np.percentile(payouts, 95)]),
            'tvar_99': np.mean(payouts[payouts >= np.percentile(payouts, 99)]),
            'prob_attachment': np.mean(payouts > 0),
            'prob_exhaustion': np.mean(payouts >= layer.limit),
            'expected_payout_ratio': np.mean(payouts) / layer.limit if layer.limit > 0 else 0,
            'coefficient_of_variation': np.std(payouts) / np.mean(payouts) if np.mean(payouts) > 0 else 0
        }
    
    def price_cat_bond(self, 
                      layer: CATBondLayer,
                      exposures: Optional[pd.DataFrame] = None,
                      risk_load: float = 0.25,
                      stress_factor: float = 1.0,
                      confidence_level: float = 0.95) -> PricingResult:
        """
        Price CAT bond with comprehensive metrics and confidence intervals.
        """
        
        # Generate losses based on model type
        if self.model_type == ModelType.EVENT_BASED:
            if exposures is None:
                raise ValueError("Exposures required for event-based modeling")
            base_losses = self.generate_event_losses(exposures)
        else:
            base_losses = self.generate_aggregate_losses()
        
        # Apply stress factor
        stressed_losses = base_losses * stress_factor
        
        # Calculate payouts
        payouts = self.calculate_layer_payout(stressed_losses, layer)
        
        # Calculate metrics
        metrics = self.calculate_comprehensive_metrics(payouts, layer)
        
        # Pricing calculations
        expected_loss = metrics['expected_loss']
        gross_premium = expected_loss * (1 + risk_load)
        fair_spread = gross_premium / layer.limit if layer.limit > 0 else 0
        
        # Bootstrap confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            payouts, confidence_level
        )
        
        return PricingResult(
            expected_loss=expected_loss,
            gross_premium=gross_premium,
            fair_spread=fair_spread,
            risk_metrics=metrics,
            confidence_intervals=confidence_intervals
        )
    
    def _calculate_confidence_intervals(self, 
                                     payouts: np.ndarray,
                                     confidence_level: float) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals."""
        n_bootstrap = 1000
        alpha = 1 - confidence_level
        
        bootstrap_means = []
        bootstrap_stds = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(payouts, size=len(payouts), replace=True)
            bootstrap_means.append(np.mean(sample))
            bootstrap_stds.append(np.std(sample))
        
        return {
            'expected_loss': (
                np.percentile(bootstrap_means, 100 * alpha/2),
                np.percentile(bootstrap_means, 100 * (1 - alpha/2))
            ),
            'std_dev': (
                np.percentile(bootstrap_stds, 100 * alpha/2),
                np.percentile(bootstrap_stds, 100 * (1 - alpha/2))
            )
        }
    
    def run_scenario_analysis(self, 
                            layer: CATBondLayer,
                            exposures: Optional[pd.DataFrame] = None,
                            scenarios: Optional[List[Dict]] = None) -> pd.DataFrame:
        """Run comprehensive scenario analysis."""
        
        if scenarios is None:
            scenarios = [
                {'name': 'Base Case', 'stress_factor': 1.0, 'risk_load': 0.25},
                {'name': 'Climate Stress', 'stress_factor': 1.2, 'risk_load': 0.30},
                {'name': 'Severe Climate', 'stress_factor': 1.5, 'risk_load': 0.40},
                {'name': 'Low Risk Load', 'stress_factor': 1.0, 'risk_load': 0.15},
                {'name': 'High Risk Load', 'stress_factor': 1.0, 'risk_load': 0.35}
            ]
        
        results = []
        for scenario in scenarios:
            result = self.price_cat_bond(
                layer=layer,
                exposures=exposures,
                risk_load=scenario['risk_load'],
                stress_factor=scenario['stress_factor']
            )
            
            results.append({
                'scenario': scenario['name'],
                'expected_loss': result.expected_loss,
                'gross_premium': result.gross_premium,
                'fair_spread': result.fair_spread,
                'var_99': result.risk_metrics['var_99'],
                'prob_attachment': result.risk_metrics['prob_attachment'],
                'prob_exhaustion': result.risk_metrics['prob_exhaustion']
            })
        
        return pd.DataFrame(results)
    
    def create_enhanced_visualization(self, 
                                    losses: np.ndarray,
                                    layers: List[CATBondLayer],
                                    save_path: Optional[str] = None) -> None:
        """Create comprehensive visualization dashboard."""
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Loss distribution with multiple layers
        ax1 = plt.subplot(2, 3, 1)
        plt.hist(losses, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        colors = ['red', 'orange', 'green', 'purple', 'brown']
        for i, layer in enumerate(layers):
            color = colors[i % len(colors)]
            plt.axvline(layer.attachment, color=color, linestyle='--', 
                       label=f'{layer.name} Attachment')
            plt.axvline(layer.exhaustion, color=color, linestyle=':', 
                       label=f'{layer.name} Exhaustion')
        plt.title('Loss Distribution with CAT Bond Layers')
        plt.xlabel('Loss ($M)')
        plt.ylabel('Frequency')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 2. Exceedance probability curves
        ax2 = plt.subplot(2, 3, 2)
        sorted_losses = np.sort(losses)[::-1]
        exceedance_probs = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
        plt.loglog(sorted_losses, exceedance_probs, 'b-', linewidth=2, label='Loss EP Curve')
        for i, layer in enumerate(layers):
            color = colors[i % len(colors)]
            plt.axvline(layer.attachment, color=color, alpha=0.7, linestyle='--')
            plt.axvline(layer.exhaustion, color=color, alpha=0.7, linestyle=':')
        plt.title('Loss Exceedance Probability')
        plt.xlabel('Loss ($M)')
        plt.ylabel('Exceedance Probability')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 3. Layer payout analysis
        ax3 = plt.subplot(2, 3, 3)
        for i, layer in enumerate(layers):
            payouts = self.calculate_layer_payout(losses, layer)
            positive_payouts = payouts[payouts > 0]
            if len(positive_payouts) > 0:
                plt.hist(positive_payouts, bins=20, alpha=0.6, 
                        color=colors[i % len(colors)], 
                        label=f'{layer.name} Payouts')
        plt.title('Layer Payout Distributions')
        plt.xlabel('Payout ($M)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Return period analysis
        ax4 = plt.subplot(2, 3, 4)
        return_periods = 1 / exceedance_probs
        plt.semilogx(return_periods, sorted_losses, 'g-', linewidth=2)
        plt.title('Loss vs Return Period')
        plt.xlabel('Return Period (Years)')
        plt.ylabel('Loss ($M)')
        plt.grid(True, alpha=0.3)
        
        # 5. Risk metrics comparison
        ax5 = plt.subplot(2, 3, 5)
        metrics_data = []
        layer_names = []
        for layer in layers:
            payouts = self.calculate_layer_payout(losses, layer)
            metrics = self.calculate_comprehensive_metrics(payouts, layer)
            metrics_data.append([
                metrics['expected_loss'],
                metrics['var_95'],
                metrics['var_99'],
                metrics['tvar_99']
            ])
            layer_names.append(layer.name or f"Layer {len(layer_names)+1}")
        
        metrics_df = pd.DataFrame(
            metrics_data, 
            columns=['Expected Loss', 'VaR 95%', 'VaR 99%', 'TVaR 99%'],
            index=layer_names
        )
        
        x = np.arange(len(layer_names))
        width = 0.2
        for i, col in enumerate(metrics_df.columns):
            plt.bar(x + i*width, metrics_df[col], width, 
                   label=col, alpha=0.8)
        
        plt.title('Risk Metrics by Layer')
        plt.xlabel('Layer')
        plt.ylabel('Value ($M)')
        plt.xticks(x + width*1.5, layer_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Pricing summary table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('tight')
        ax6.axis('off')
        
        pricing_data = []
        for layer in layers:
            result = self.price_cat_bond(layer)
            pricing_data.append([
                layer.name or f"Layer {len(pricing_data)+1}",
                f"${layer.attachment:.0f}M xs ${layer.limit:.0f}M",
                f"${result.expected_loss:.2f}M",
                f"{result.fair_spread:.2%}",
                f"{result.risk_metrics['prob_attachment']:.1%}",
                f"{result.risk_metrics['prob_exhaustion']:.1%}"
            ])
        
        table = ax6.table(
            cellText=pricing_data,
            colLabels=['Layer', 'Structure', 'Expected Loss', 'Fair Spread', 'P(Attach)', 'P(Exhaust)'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax6.set_title('Pricing Summary', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Example usage with enhanced features
def run_enhanced_analysis():
    """Run comprehensive CAT bond analysis."""
    
    # Initialize pricer
    pricer = EnhancedCATBondPricer(ModelType.AGGREGATE, random_seed=42)
    
    # Define multiple layers
    layers = [
        CATBondLayer(attachment=100, limit=50, name="Layer 1"),
        CATBondLayer(attachment=150, limit=50, name="Layer 2"), 
        CATBondLayer(attachment=200, limit=100, name="Layer 3")
    ]
    
    # Generate losses
    losses = pricer.generate_aggregate_losses(num_simulations=25000)
    
    # Run scenario analysis
    scenario_results = pricer.run_scenario_analysis(layers[0])
    print("Scenario Analysis Results:")
    print(scenario_results.round(3))
    
    # Price individual layer with confidence intervals
    result = pricer.price_cat_bond(layers[0])
    print(f"\nLayer 1 Pricing:")
    print(f"Expected Loss: ${result.expected_loss:.2f}M")
    print(f"Fair Spread: {result.fair_spread:.2%}")
    print(f"95% CI for Expected Loss: ${result.confidence_intervals['expected_loss'][0]:.2f}M - ${result.confidence_intervals['expected_loss'][1]:.2f}M")
    
    # Create enhanced visualization
    pricer.create_enhanced_visualization(losses, layers)
    
    return pricer, losses, layers

# Uncomment to run the enhanced analysis
# pricer, losses, layers = run_enhanced_analysis()
```## 📈 **Performance Optimizations**

### **1. Vectorized Loss Calculations**
```python
def vectorized_loss_calculation(exposures: pd.DataFrame, 
                              footprints: pd.DataFrame,
                              vuln_curves: Dict,
                              stress_factor: float = 1.0) -> pd.Series:
    """Optimized vectorized loss calculation."""
    
    # Merge exposures with footprints for vectorized operations
    merged = exposures.merge(footprints, on='area_peril_id', how='inner')
    
    # Vectorized stress application
    merged['stressed_intensity'] = np.minimum(
        10, (merged['intensity_bin'] * stress_factor).astype(int)
    )
    
    # Vectorized vulnerability lookup
    construction_types = merged['construction_type'].fillna('wood_frame')
    damage_ratios = []
    
    for idx, row in merged.iterrows():
        curve = vuln_curves.get(row['construction_type'], vuln_curves['wood_frame'])
        damage_ratios.append(curve.get(row['stressed_intensity'], 0))
    
    merged['damage_ratio'] = damage_ratios
    merged['loss'] = merged['tiv'] * merged['damage_ratio']
    
    # Group by event to get total losses
    return merged.groupby('event_id')['loss'].sum()

def parallel_simulation(pricer, layer, n_simulations=1000, n_cores=4):
    """Run simulations in parallel for faster computation."""
    from multiprocessing import Pool
    import functools
    
    def simulate_chunk(chunk_size):
        """Simulate a chunk of events."""
        local_pricer = EnhancedCATBondPricer(random_seed=np.random.randint(10000))
        losses = local_pricer.generate_aggregate_losses(chunk_size)
        return local_pricer.calculate_layer_payout(losses, layer)
    
    chunk_size = n_simulations // n_cores
    
    with Pool(n_cores) as pool:
        results = pool.map(simulate_chunk, [chunk_size] * n_cores)
    
    return np.concatenate(results)
