import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging

class CATBondPricer:
    """Enhanced CAT Bond pricing model with improved structure and validation."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
        
    def create_spatial_grid(self, lat_range: Tuple[float, float], 
                          lon_range: Tuple[float, float], 
                          grid_size: Tuple[int, int]) -> pd.DataFrame:
        """Create spatial grid for area perils with validation."""
        if lat_range[0] >= lat_range[1] or lon_range[0] >= lon_range[1]:
            raise ValueError("Invalid coordinate ranges")
            
        lat_points = np.linspace(lat_range[0], lat_range[1], grid_size[0])
        lon_points = np.linspace(lon_range[0], lon_range[1], grid_size[1])
        
        area_perils = []
        area_peril_id = 1
        for lat in lat_points:
            for lon in lon_points:
                area_perils.append({
                    'area_peril_id': area_peril_id, 
                    'lat': lat, 
                    'lon': lon
                })
                area_peril_id += 1
                
        return pd.DataFrame(area_perils)
    
    def generate_exposures(self, area_perils: pd.DataFrame, 
                         num_locations: int,
                         tiv_distribution: Dict) -> pd.DataFrame:
        """Generate synthetic exposures with configurable parameters."""
        if num_locations <= 0:
            raise ValueError("Number of locations must be positive")
            
        exposures = []
        for loc_id in range(1, num_locations + 1):
            # Geographic clustering logic
            if np.random.rand() < tiv_distribution.get('cluster_prob', 0.7):
                area_subset = area_perils[
                    area_perils['area_peril_id'] <= len(area_perils) // 2
                ]
            else:
                area_subset = area_perils[
                    area_perils['area_peril_id'] > len(area_perils) // 2
                ]
            
            chosen = area_subset.sample(1).iloc[0]
            
            # Enhanced TIV generation
            tiv = np.random.choice([
                np.random.uniform(
                    tiv_distribution.get('low_min', 0.5e6),
                    tiv_distribution.get('low_max', 2e6)
                ),
                np.random.uniform(
                    tiv_distribution.get('high_min', 5e6),
                    tiv_distribution.get('high_max', 10e6)
                )
            ])
            
            exposures.append({
                'loc_id': loc_id,
                'area_peril_id': chosen['area_peril_id'],
                'vulnerability_id': 1,
                'tiv': tiv
            })
            
        return pd.DataFrame(exposures)
    
    def create_vulnerability_curve(self, curve_type: str = 'power') -> Dict[int, float]:
        """Create vulnerability curves with different damage functions."""
        curves = {
            'power': {i: min(1.0, 0.1 * i**1.5) for i in range(1, 11)},
            'linear': {i: min(1.0, 0.1 * i) for i in range(1, 11)},
            'exponential': {i: min(1.0, 0.05 * (1.5**i)) for i in range(1, 11)}
        }
        
        if curve_type not in curves:
            raise ValueError(f"Unknown curve type: {curve_type}")
            
        return curves[curve_type]
    
    def generate_event_footprints(self, area_perils: pd.DataFrame, 
                                num_events: int,
                                footprint_size: int = 10) -> pd.DataFrame:
        """Generate synthetic event footprints with validation."""
        if num_events <= 0 or footprint_size <= 0:
            raise ValueError("Events and footprint size must be positive")
            
        footprints = []
        for event_id in range(1, num_events + 1):
            # Ensure we don't try to sample more than available
            sample_size = min(footprint_size, len(area_perils))
            affected_ids = np.random.choice(
                area_perils['area_peril_id'], 
                size=sample_size, 
                replace=False
            )
            
            for aid in affected_ids:
                intensity_bin = np.random.randint(1, 11)
                footprints.append({
                    'event_id': event_id,
                    'area_peril_id': aid,
                    'intensity_bin': intensity_bin
                })
                
        return pd.DataFrame(footprints)
    
    def calculate_metrics(self, losses: np.ndarray, 
                        attachment: float, 
                        limit: float) -> Dict[str, float]:
        """Calculate comprehensive CAT bond metrics."""
        payouts = np.maximum(np.minimum(losses - attachment, limit), 0)
        
        return {
            'expected_loss': np.mean(payouts),
            'std_dev': np.std(payouts),
            'max_loss': np.max(payouts),
            'var_95': np.percentile(payouts, 95),
            'var_99': np.percentile(payouts, 99),
            'prob_attachment': np.mean(losses > attachment),
            'prob_exhaustion': np.mean(losses > (attachment + limit)),
            'expected_payout_ratio': np.mean(payouts) / limit if limit > 0 else 0
        }
    
    def price_cat_bond(self, area_perils: pd.DataFrame,
                      exposures: pd.DataFrame,
                      footprints: pd.DataFrame,
                      vulnerability_curve: Dict[int, float],
                      attachment: float,
                      limit: float,
                      risk_load: float = 0.25,
                      stress_factor: float = 1.0,
                      visualize: bool = True) -> Dict:
        """Enhanced CAT bond pricing with comprehensive metrics."""
        
        # Input validation
        if attachment < 0 or limit <= 0 or risk_load < 0:
            raise ValueError("Invalid pricing parameters")
        
        # Calculate losses for each event
        event_losses = []
        unique_events = footprints['event_id'].unique()
        
        for event_id in unique_events:
            total_loss = 0.0
            affected_areas = footprints[footprints['event_id'] == event_id]
            
            for _, exposure in exposures.iterrows():
                matching_footprint = affected_areas[
                    affected_areas['area_peril_id'] == exposure['area_peril_id']
                ]
                
                if not matching_footprint.empty:
                    intensity = matching_footprint.iloc[0]['intensity_bin']
                    stressed_intensity = min(10, int(intensity * stress_factor))
                    damage_ratio = vulnerability_curve[stressed_intensity]
                    total_loss += exposure['tiv'] * damage_ratio
                    
            event_losses.append(total_loss)
        
        losses_array = np.array(event_losses)
        
        # Calculate comprehensive metrics
        metrics = self.calculate_metrics(losses_array, attachment, limit)
        
        # Pricing calculations
        gross_premium = metrics['expected_loss'] * (1 + risk_load)
        fair_spread = gross_premium / limit if limit > 0 else 0
        
        # Enhanced visualization
        if visualize:
            self._create_enhanced_visualization(
                losses_array, attachment, limit, metrics
            )
        
        return {
            'expected_loss': metrics['expected_loss'],
            'gross_premium': gross_premium,
            'fair_spread': fair_spread,
            'risk_metrics': metrics,
            'losses': losses_array
        }
    
    def _create_enhanced_visualization(self, losses: np.ndarray, 
                                     attachment: float, 
                                     limit: float,
                                     metrics: Dict) -> None:
        """Create comprehensive visualization dashboard."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss distribution histogram
        ax1.hist(losses, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(attachment, color='orange', linestyle='--', 
                   label=f'Attachment: ${attachment/1e6:.1f}M')
        ax1.axvline(attachment + limit, color='red', linestyle='--', 
                   label=f'Exhaustion: ${(attachment+limit)/1e6:.1f}M')
        ax1.set_title('Event Loss Distribution')
        ax1.set_xlabel('Loss ($)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Exceedance probability curve
        sorted_losses = np.sort(losses)[::-1]
        exceedance_probs = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
        ax2.loglog(sorted_losses, exceedance_probs, 'b-', linewidth=2)
        ax2.axvline(attachment, color='orange', linestyle='--', alpha=0.7)
        ax2.axvline(attachment + limit, color='red', linestyle='--', alpha=0.7)
        ax2.set_title('Loss Exceedance Probability')
        ax2.set_xlabel('Loss ($)')
        ax2.set_ylabel('Exceedance Probability')
        ax2.grid(True, alpha=0.3)
        
        # Payout distribution
        payouts = np.maximum(np.minimum(losses - attachment, limit), 0)
        ax3.hist(payouts[payouts > 0], bins=20, color='lightcoral', 
                edgecolor='black', alpha=0.7)
        ax3.set_title('CAT Bond Payout Distribution')
        ax3.set_xlabel('Payout ($)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Risk metrics summary
        ax4.axis('off')
        metrics_text = f"""
        Expected Loss: ${metrics['expected_loss']/1e6:.2f}M
        Standard Deviation: ${metrics['std_dev']/1e6:.2f}M
        VaR 95%: ${metrics['var_95']/1e6:.2f}M
        VaR 99%: ${metrics['var_99']/1e6:.2f}M
        Max Loss: ${metrics['max_loss']/1e6:.2f}M
        
        Prob. Attachment: {metrics['prob_attachment']:.2%}
        Prob. Exhaustion: {metrics['prob_exhaustion']:.2%}
        Expected Payout Ratio: {metrics['expected_payout_ratio']:.2%}
        """
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax4.set_title('Risk Metrics Summary')
        
        plt.tight_layout()
        plt.show()

# Example usage with improved interface
def run_enhanced_cat_bond_analysis():
    """Run enhanced CAT bond analysis with improved parameters."""
    
    pricer = CATBondPricer(random_seed=42)
    
    # Create spatial grid (Florida region)
    area_perils = pricer.create_spatial_grid(
        lat_range=(25.0, 27.0),
        lon_range=(-82.0, -80.0),
        grid_size=(5, 5)
    )
    
    # Generate exposures with custom distribution
    tiv_config = {
        'cluster_prob': 0.7,
        'low_min': 0.5e6,
        'low_max': 2e6,
        'high_min': 5e6,
        'high_max': 10e6
    }
    
    exposures = pricer.generate_exposures(
        area_perils=area_perils,
        num_locations=100,
        tiv_distribution=tiv_config
    )
    
    # Create vulnerability curve
    vuln_curve = pricer.create_vulnerability_curve('power')
    
    # Generate event footprints
    footprints = pricer.generate_event_footprints(
        area_perils=area_perils,
        num_events=1000,  # Increased for better statistics
        footprint_size=10
    )
    
    # Price CAT bond
    results = pricer.price_cat_bond(
        area_perils=area_perils,
        exposures=exposures,
        footprints=footprints,
        vulnerability_curve=vuln_curve,
        attachment=5e6,
        limit=10e6,
        risk_load=0.25,
        stress_factor=1.0,
        visualize=True
    )
    
    print(f"\n=== CAT BOND PRICING RESULTS ===")
    print(f"Expected Loss: ${results['expected_loss']:,.2f}")
    print(f"Gross Premium: ${results['gross_premium']:,.2f}")
    print(f"Fair Spread: {results['fair_spread']:.2%}")
    print(f"VaR 99%: ${results['risk_metrics']['var_99']:,.2f}")
    
    return results

# Run the analysis
# results = run_enhanced_cat_bond_analysis()
