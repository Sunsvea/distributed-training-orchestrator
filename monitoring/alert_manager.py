import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import threading

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics for alerting"""
    SYSTEM = "system"
    TRAINING = "training"
    NETWORK = "network"
    CUSTOM = "custom"

@dataclass
class Alert:
    """Represents an alert instance"""
    name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    resolved: bool = False
    resolved_timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlertRule:
    """Defines an alert rule"""
    name: str
    metric_type: MetricType
    metric_name: str
    condition: str  # "greater_than", "less_than", "equals", "not_equals"
    threshold: float
    severity: AlertSeverity
    duration: float = 60.0  # How long condition must be true before alerting
    message_template: str = ""
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class AlertManager:
    """Manages alerting rules and active alerts"""
    
    def __init__(self):
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.condition_start_times: Dict[str, float] = {}
        
        # Thread safety
        self.alert_lock = threading.Lock()
        
        # Configuration
        self.max_history_size = 1000
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        with self.alert_lock:
            self.alert_rules.append(rule)
        
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str) -> bool:
        """Remove an alert rule by name"""
        with self.alert_lock:
            for i, rule in enumerate(self.alert_rules):
                if rule.name == rule_name:
                    del self.alert_rules[i]
                    logger.info(f"Removed alert rule: {rule_name}")
                    return True
        
        return False
    
    def evaluate_alert_rules(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate all alert rules against current metrics"""
        triggered_alerts = []
        current_time = time.time()
        
        with self.alert_lock:
            for rule in self.alert_rules:
                if not rule.enabled:
                    continue
                
                # Get metric value
                metric_value = self._get_metric_value(metrics, rule.metric_type, rule.metric_name)
                if metric_value is None:
                    continue
                
                # Check condition
                condition_met = self._evaluate_condition(metric_value, rule.condition, rule.threshold)
                
                if condition_met:
                    # Check if condition has been met for required duration
                    condition_key = f"{rule.name}_{rule.metric_name}"
                    
                    if condition_key not in self.condition_start_times:
                        self.condition_start_times[condition_key] = current_time
                    
                    duration = current_time - self.condition_start_times[condition_key]
                    
                    if duration >= rule.duration:
                        # Check if alert is already active
                        if rule.name not in self.active_alerts:
                            alert = self._create_alert(rule, metric_value)
                            self.active_alerts[rule.name] = alert
                            self.alert_history.append(alert)
                            triggered_alerts.append(alert)
                            
                            logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
                else:
                    # Condition not met, remove from start times and resolve alert if active
                    condition_key = f"{rule.name}_{rule.metric_name}"
                    if condition_key in self.condition_start_times:
                        del self.condition_start_times[condition_key]
                    
                    if rule.name in self.active_alerts:
                        alert = self.active_alerts[rule.name]
                        alert.resolved = True
                        alert.resolved_timestamp = current_time
                        del self.active_alerts[rule.name]
                        
                        logger.info(f"Alert resolved: {alert.name}")
            
            # Cleanup old history
            if len(self.alert_history) > self.max_history_size:
                self.alert_history = self.alert_history[-self.max_history_size:]
        
        return triggered_alerts
    
    def _get_metric_value(self, metrics: Dict[str, Any], metric_type: MetricType, metric_name: str) -> Optional[float]:
        """Extract metric value from metrics dict"""
        try:
            if metric_type == MetricType.SYSTEM:
                if "system" in metrics and metrics["system"]:
                    return getattr(metrics["system"], metric_name, None)
            elif metric_type == MetricType.TRAINING:
                if "training" in metrics and metrics["training"]:
                    return getattr(metrics["training"], metric_name, None)
            elif metric_type == MetricType.NETWORK:
                if "network" in metrics and metrics["network"]:
                    return getattr(metrics["network"], metric_name, None)
            elif metric_type == MetricType.CUSTOM:
                if "custom" in metrics:
                    return metrics["custom"].get(metric_name, None)
        except Exception as e:
            logger.error(f"Error getting metric value {metric_name}: {e}")
        
        return None
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate a condition against a threshold"""
        if condition == "greater_than":
            return value > threshold
        elif condition == "less_than":
            return value < threshold
        elif condition == "equals":
            return abs(value - threshold) < 1e-6
        elif condition == "not_equals":
            return abs(value - threshold) >= 1e-6
        elif condition == "greater_than_or_equal":
            return value >= threshold
        elif condition == "less_than_or_equal":
            return value <= threshold
        else:
            logger.warning(f"Unknown condition: {condition}")
            return False
    
    def _create_alert(self, rule: AlertRule, metric_value: float) -> Alert:
        """Create an alert from a rule"""
        message = rule.message_template or f"{rule.metric_name} is {metric_value} (threshold: {rule.threshold})"
        
        return Alert(
            name=rule.name,
            severity=rule.severity,
            message=message,
            timestamp=time.time(),
            metadata={
                "metric_type": rule.metric_type.value,
                "metric_name": rule.metric_name,
                "metric_value": metric_value,
                "threshold": rule.threshold,
                "condition": rule.condition
            }
        )
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active alerts"""
        with self.alert_lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: Optional[int] = None, 
                         active_only: bool = False) -> List[Alert]:
        """Get alert history with optional filtering"""
        with self.alert_lock:
            history = self.alert_history.copy()
        
        # Filter by active status
        if active_only:
            history = [alert for alert in history if not alert.resolved]
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Limit results
        if limit:
            history = history[:limit]
        
        return history
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status"""
        with self.alert_lock:
            active_by_severity = {}
            for alert in self.active_alerts.values():
                severity = alert.severity.value
                if severity not in active_by_severity:
                    active_by_severity[severity] = 0
                active_by_severity[severity] += 1
            
            # Recent alerts (last hour)
            recent_alerts = [
                alert for alert in self.alert_history
                if alert.timestamp > time.time() - 3600
            ]
            
            return {
                "active_alerts_count": len(self.active_alerts),
                "active_by_severity": active_by_severity,
                "recent_alerts_count": len(recent_alerts),
                "total_rules": len(self.alert_rules),
                "enabled_rules": len([r for r in self.alert_rules if r.enabled])
            }
    
    def acknowledge_alert(self, alert_name: str) -> bool:
        """Acknowledge an active alert"""
        with self.alert_lock:
            if alert_name in self.active_alerts:
                alert = self.active_alerts[alert_name]
                alert.metadata["acknowledged"] = True
                alert.metadata["acknowledged_at"] = time.time()
                logger.info(f"Alert acknowledged: {alert_name}")
                return True
        
        return False
    
    def silence_alert(self, alert_name: str, duration: float = 3600.0) -> bool:
        """Silence an alert for a specified duration"""
        with self.alert_lock:
            if alert_name in self.active_alerts:
                alert = self.active_alerts[alert_name]
                alert.metadata["silenced"] = True
                alert.metadata["silenced_until"] = time.time() + duration
                logger.info(f"Alert silenced: {alert_name} for {duration} seconds")
                return True
        
        return False
    
    def clear_alert_history(self):
        """Clear alert history"""
        with self.alert_lock:
            self.alert_history.clear()
        
        logger.info("Alert history cleared")
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable an alert rule"""
        with self.alert_lock:
            for rule in self.alert_rules:
                if rule.name == rule_name:
                    rule.enabled = True
                    logger.info(f"Alert rule enabled: {rule_name}")
                    return True
        
        return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable an alert rule"""
        with self.alert_lock:
            for rule in self.alert_rules:
                if rule.name == rule_name:
                    rule.enabled = False
                    logger.info(f"Alert rule disabled: {rule_name}")
                    return True
        
        return False
    
    def get_rule_status(self) -> List[Dict[str, Any]]:
        """Get status of all alert rules"""
        with self.alert_lock:
            return [
                {
                    "name": rule.name,
                    "enabled": rule.enabled,
                    "metric_type": rule.metric_type.value,
                    "metric_name": rule.metric_name,
                    "condition": rule.condition,
                    "threshold": rule.threshold,
                    "severity": rule.severity.value,
                    "duration": rule.duration
                }
                for rule in self.alert_rules
            ]