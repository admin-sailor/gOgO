/**
 * Chart utilities for BTTS dashboard
 * Uses canvas API for pie, donut, and bar charts
 */

class ChartRenderer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.width = this.canvas ? this.canvas.width : 0;
        this.height = this.canvas ? this.canvas.height : 0;

        // Color palette matching dark theme
        this.colors = {
            green: '#48bb78',
            red: '#f56565',
            yellow: '#ecc94b',
            purple: '#9f7aea',
            blue: '#4299e1',
            gray: '#718096',
        };
    }

    /**
     * Draw pie chart
     * data: { labels: [], values: [], colors: [] }
     */
    drawPieChart(data) {
        if (!this.ctx) return;

        const centerX = this.width / 2;
        const centerY = this.height / 2;
        const radius = Math.min(this.width, this.height) / 2 - 20;

        const total = data.values.reduce((a, b) => a + b, 0);
        let currentAngle = -Math.PI / 2;

        // Draw pie slices
        data.values.forEach((value, index) => {
            const sliceAngle = (value / total) * 2 * Math.PI;

            // Draw slice
            this.ctx.beginPath();
            this.ctx.moveTo(centerX, centerY);
            this.ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
            this.ctx.closePath();
            this.ctx.fillStyle = data.colors[index] || this.colors.blue;
            this.ctx.fill();

            // Draw border
            this.ctx.strokeStyle = '#1a1f2e';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();

            // Draw label
            const labelAngle = currentAngle + sliceAngle / 2;
            const labelX = centerX + Math.cos(labelAngle) * (radius * 0.7);
            const labelY = centerY + Math.sin(labelAngle) * (radius * 0.7);

            this.ctx.fillStyle = '#e2e8f0';
            this.ctx.font = '12px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(`${((value / total) * 100).toFixed(1)}%`, labelX, labelY);

            currentAngle += sliceAngle;
        });

        // Draw legend
        this.drawLegend(data.labels, data.colors, 10);
    }

    /**
     * Draw donut chart
     */
    drawDonutChart(data) {
        if (!this.ctx) return;

        const centerX = this.width / 2;
        const centerY = this.height / 2;
        const outerRadius = Math.min(this.width, this.height) / 2 - 20;
        const innerRadius = outerRadius * 0.6;

        const total = data.values.reduce((a, b) => a + b, 0);
        let currentAngle = -Math.PI / 2;

        data.values.forEach((value, index) => {
            const sliceAngle = (value / total) * 2 * Math.PI;

            // Draw outer arc
            this.ctx.beginPath();
            this.ctx.arc(centerX, centerY, outerRadius, currentAngle, currentAngle + sliceAngle);
            this.ctx.lineTo(
                centerX + Math.cos(currentAngle + sliceAngle) * innerRadius,
                centerY + Math.sin(currentAngle + sliceAngle) * innerRadius
            );
            this.ctx.arc(centerX, centerY, innerRadius, currentAngle + sliceAngle, currentAngle, true);
            this.ctx.closePath();

            this.ctx.fillStyle = data.colors[index] || this.colors.blue;
            this.ctx.fill();

            this.ctx.strokeStyle = '#1a1f2e';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();

            currentAngle += sliceAngle;
        });

        // Draw center label
        this.ctx.fillStyle = '#e2e8f0';
        this.ctx.font = 'bold 16px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(data.centerLabel || 'Data', centerX, centerY);
    }

    /**
     * Draw horizontal bar chart
     */
    drawBarChart(data) {
        if (!this.ctx) return;

        const padding = 40;
        const chartWidth = this.width - padding * 2;
        const chartHeight = this.height - padding * 2;
        const barHeight = chartHeight / data.labels.length;
        const maxValue = Math.max(...data.values);

        // Draw background
        this.ctx.fillStyle = '#252d3d';
        this.ctx.fillRect(padding, padding, chartWidth, chartHeight);

        // Draw grid lines
        this.ctx.strokeStyle = '#374151';
        this.ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {
            const x = padding + (chartWidth / 5) * i;
            this.ctx.beginPath();
            this.ctx.moveTo(x, padding);
            this.ctx.lineTo(x, padding + chartHeight);
            this.ctx.stroke();
        }

        // Draw bars
        data.values.forEach((value, index) => {
            const barWidth = (value / maxValue) * (chartWidth - 60);
            const y = padding + index * barHeight + barHeight / 2 - 10;

            this.ctx.fillStyle = data.colors[index] || this.colors.blue;
            this.ctx.fillRect(padding + 40, y, barWidth, 20);

            // Label
            this.ctx.fillStyle = '#a0aec0';
            this.ctx.font = '12px Arial';
            this.ctx.textAlign = 'right';
            this.ctx.fillText(data.labels[index], padding + 35, y + 15);

            // Value
            this.ctx.fillStyle = '#e2e8f0';
            this.ctx.textAlign = 'left';
            this.ctx.fillText(value.toFixed(2), padding + barWidth + 45, y + 15);
        });

        // Draw axes
        this.ctx.strokeStyle = '#718096';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(padding + 40, padding);
        this.ctx.lineTo(padding + 40, padding + chartHeight);
        this.ctx.lineTo(padding + chartWidth, padding + chartHeight);
        this.ctx.stroke();
    }

    /**
     * Draw legend
     */
    drawLegend(labels, colors, startY) {
        let y = startY;
        const x = this.width - 150;

        this.ctx.font = '12px Arial';
        this.ctx.fillStyle = '#a0aec0';

        labels.forEach((label, index) => {
            // Color box
            this.ctx.fillStyle = colors[index] || this.colors.blue;
            this.ctx.fillRect(x, y, 12, 12);

            // Label
            this.ctx.fillStyle = '#a0aec0';
            this.ctx.textAlign = 'left';
            this.ctx.fillText(label, x + 18, y + 10);

            y += 20;
        });
    }

    /**
     * Clear canvas
     */
    clear() {
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.width, this.height);
        }
    }
}

/**
 * Dashboard data aggregator
 */
class DashboardData {
    constructor() {
        this.predictions = [];
        this.stats = {};
    }

    addPrediction(prediction) {
        this.predictions.push({
            ...prediction,
            timestamp: new Date(),
        });
    }

    getPredictionStats() {
        if (this.predictions.length === 0) {
            return {
                total: 0,
                btts_yes: 0,
                btts_no: 0,
                avg_probability: 0,
                hit_rate: 0,
            };
        }

        const bttsYes = this.predictions.filter(p => p.btts_prediction).length;
        const avgProb = this.predictions.reduce((sum, p) => sum + p.btts_probability, 0) / this.predictions.length;

        return {
            total: this.predictions.length,
            btts_yes: bttsYes,
            btts_no: this.predictions.length - bttsYes,
            avg_probability: avgProb,
            hit_rate: ((bttsYes / this.predictions.length) * 100).toFixed(1),
        };
    }

    getModelComparison() {
        const models = {
            ensemble: [],
            logistic_regression: [],
            neural_network: [],
        };

        this.predictions.forEach(p => {
            if (p.model) {
                const model = p.model.toLowerCase().replace(' ', '_');
                if (models[model]) {
                    models[model].push(p.btts_probability);
                }
            }
        });

        return {
            ensemble: models.ensemble.length > 0 ? 
                (models.ensemble.reduce((a, b) => a + b, 0) / models.ensemble.length * 100).toFixed(1) : 'N/A',
            logistic_regression: models.logistic_regression.length > 0 ? 
                (models.logistic_regression.reduce((a, b) => a + b, 0) / models.logistic_regression.length * 100).toFixed(1) : 'N/A',
            neural_network: models.neural_network.length > 0 ? 
                (models.neural_network.reduce((a, b) => a + b, 0) / models.neural_network.length * 100).toFixed(1) : 'N/A',
        };
    }

    getProbabilityDistribution() {
        const ranges = {
            '0-20%': 0,
            '20-40%': 0,
            '40-60%': 0,
            '60-80%': 0,
            '80-100%': 0,
        };

        this.predictions.forEach(p => {
            const prob = p.btts_probability * 100;
            if (prob < 20) ranges['0-20%']++;
            else if (prob < 40) ranges['20-40%']++;
            else if (prob < 60) ranges['40-60%']++;
            else if (prob < 80) ranges['60-80%']++;
            else ranges['80-100%']++;
        });

        return ranges;
    }
}

// Expose to window for browser
if (typeof window !== 'undefined') {
    window.ChartRenderer = ChartRenderer;
    window.DashboardData = DashboardData;
}

// Export for use in Node.js/CommonJS
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ChartRenderer, DashboardData };
}
