<script setup>
import { ref, computed, watch } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart, ScatterChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([LineChart, ScatterChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer])

const props = defineProps({
  type: { type: String, default: 'poly6' },
})

// f(x) = x^6 - 3x^4 + 2x^3 + x
function poly6(x) {
  return x ** 6 - 3 * x ** 4 + 2 * x ** 3 + x
}

// Derivatives of poly6 evaluated at point a
function poly6Coeffs(a) {
  const a2 = a * a, a3 = a2 * a, a4 = a3 * a, a5 = a4 * a, a6 = a5 * a
  return [
    a6 - 3 * a4 + 2 * a3 + a,                          // f(a)
    (6 * a5 - 12 * a3 + 6 * a2 + 1),                    // f'(a)
    (30 * a4 - 36 * a2 + 12 * a) / 2,                   // f''(a)/2!
    (120 * a3 - 72 * a + 12) / 6,                        // f'''(a)/3!
    (360 * a2 - 72) / 24,                                // f''''(a)/4!
    (720 * a) / 120,                                     // f'''''(a)/5!
    720 / 720,                                            // f''''''(a)/6!
  ]
}

// ln(x) coefficients at point a
function lnCoeffs(a) {
  return [
    Math.log(a),
    ...Array.from({ length: 20 }, (_, i) => {
      const n = i + 1
      return ((-1) ** (n + 1)) / (n * a ** n)
    }),
  ]
}

// e^x coefficients at point a: f^(n)(a)/n! = e^a / n!
function expCoeffs(a) {
  const ea = Math.exp(a)
  let factorial = 1
  return Array.from({ length: 20 }, (_, i) => {
    if (i > 0) factorial *= i
    return ea / factorial
  })
}

// sin(x) coefficients at point a: f^(n)(a)/n!
// Derivatives cycle: sin, cos, -sin, -cos
function sinCoeffs(a) {
  let factorial = 1
  return Array.from({ length: 20 }, (_, i) => {
    if (i > 0) factorial *= i
    const phase = a + (i * Math.PI) / 2
    return Math.sin(phase) / factorial
  })
}

const configs = {
  poly6: {
    label: 'f(x) = x⁶ − 3x⁴ + 2x³ + x',
    f: poly6,
    getCoeffs: poly6Coeffs,
    defaultA: 0,
    aMin: -2,
    aMax: 2,
    aStep: 0.1,
    xMin: -3, xMax: 3,
    yMin: -20, yMax: 40,
    maxOrder: 8,
  },
  ln: {
    label: 'f(x) = ln(x)',
    f: x => Math.log(x),
    getCoeffs: lnCoeffs,
    defaultA: 1,
    aMin: 0.2,
    aMax: 4,
    aStep: 0.1,
    xMin: 0.05, xMax: 5,
    yMin: -4, yMax: 3,
    maxOrder: 12,
  },
  exp: {
    label: 'f(x) = eˣ',
    f: x => Math.exp(x),
    getCoeffs: expCoeffs,
    defaultA: 0,
    aMin: -3,
    aMax: 3,
    aStep: 0.1,
    xMin: -4, xMax: 4,
    yMin: -2, yMax: 30,
    maxOrder: 10,
  },
  sin: {
    label: 'f(x) = sin(x)',
    f: x => Math.sin(x),
    getCoeffs: sinCoeffs,
    defaultA: 0,
    aMin: -3.1,
    aMax: 3.1,
    aStep: 0.1,
    xMin: -4 * Math.PI, xMax: 4 * Math.PI,
    yMin: -3, yMax: 3,
    maxOrder: 15,
  },
}

const config = computed(() => configs[props.type])
const n = ref(1)
const a = ref(config.value.defaultA)

watch(() => props.type, () => {
  a.value = config.value.defaultA
  n.value = 1
})

const coeffs = computed(() => config.value.getCoeffs(a.value))

const step = 0.02
const xVals = computed(() => {
  const { xMin, xMax } = config.value
  const vals = []
  for (let x = xMin; x <= xMax; x += step) vals.push(+x.toFixed(4))
  return vals
})

function taylorPoly(x, order) {
  const dx = x - a.value
  let result = 0
  for (let i = 0; i <= order && i < coeffs.value.length; i++) {
    result += coeffs.value[i] * dx ** i
  }
  return result
}

// Build LaTeX formula string for the Taylor polynomial
const taylorFormula = computed(() => {
  const c = coeffs.value
  const order = n.value
  const aVal = a.value
  const terms = []

  for (let i = 0; i <= order && i < c.length; i++) {
    const coeff = c[i]
    if (Math.abs(coeff) < 1e-10) continue

    const sign = coeff >= 0 ? (terms.length > 0 ? ' + ' : '') : (terms.length > 0 ? ' - ' : '-')
    const absCoeff = Math.abs(coeff)

    let coeffStr
    if (i === 0) {
      coeffStr = absCoeff < 1e-10 ? '0' : (Number.isInteger(absCoeff) ? String(absCoeff) : absCoeff.toFixed(2))
    } else {
      coeffStr = Math.abs(absCoeff - 1) < 1e-10 ? '' : (Number.isInteger(absCoeff) ? String(absCoeff) : absCoeff.toFixed(2))
    }

    let xPart
    if (i === 0) {
      xPart = ''
    } else if (i === 1) {
      xPart = aVal === 0 ? 'x' : `(x{-}${Math.abs(aVal).toFixed(1)})`
    } else {
      xPart = aVal === 0 ? `x^{${i}}` : `(x{-}${Math.abs(aVal).toFixed(1)})^{${i}}`
    }

    // Fix sign display for negative a
    if (aVal > 0 && i > 0) {
      xPart = aVal === 0 ? (i === 1 ? 'x' : `x^{${i}}`) : `(x{-}${aVal.toFixed(1)})${i > 1 ? `^{${i}}` : ''}`
    } else if (aVal < 0 && i > 0) {
      xPart = `(x{+}${Math.abs(aVal).toFixed(1)})${i > 1 ? `^{${i}}` : ''}`
    } else if (aVal === 0 && i > 0) {
      xPart = i === 1 ? 'x' : `x^{${i}}`
    }

    terms.push(`${sign}${coeffStr}${xPart}`)
  }

  const formula = terms.join('') || '0'
  return `T_{${order}}(x) = ${formula}`
})

const option = computed(() => {
  const { label, f, xMin, xMax, yMin, yMax } = config.value
  return {
    animation: false,
    grid: { left: 50, right: 20, top: 36, bottom: 36 },
    xAxis: {
      type: 'value', min: xMin, max: xMax,
      splitLine: { lineStyle: { color: '#333' } },
      axisLine: { lineStyle: { color: '#666' } },
      axisLabel: { color: '#aaa' },
    },
    yAxis: {
      type: 'value', min: yMin, max: yMax,
      splitLine: { lineStyle: { color: '#333' } },
      axisLine: { lineStyle: { color: '#666' } },
      axisLabel: { color: '#aaa' },
    },
    legend: {
      data: [label, `T${'{'}${n.value}{'}'}(x)`],
      textStyle: { color: '#aaa', fontSize: 11 },
      top: 6,
    },
    series: [
      {
        name: label,
        type: 'line',
        data: xVals.value.map(x => [x, f(x)]),
        showSymbol: false,
        lineStyle: { width: 2, color: '#60a5fa' },
        smooth: true,
      },
      {
        name: `T${'{'}${n.value}{'}'}(x)`,
        type: 'line',
        data: xVals.value.map(x => [x, taylorPoly(x, n.value)]),
        showSymbol: false,
        lineStyle: { width: 2, color: '#f97316' },
        smooth: true,
      },
      {
        type: 'scatter',
        data: [[a.value, f(a.value)]],
        symbolSize: 10,
        itemStyle: { color: '#f97316', borderColor: '#fff', borderWidth: 1 },
      },
    ],
  }
})
</script>

<template>
  <div class="flex flex-col w-full h-full">
    <!-- Control bar -->
    <div class="control-bar flex items-center gap-5 py-2 px-4 rounded-lg">
      <span class="text-xs text-gray-400 whitespace-nowrap">{{ config.label }}</span>

      <!-- Expansion point a -->
      <div class="flex items-center gap-2">
        <span class="slider-label-a">a</span>
        <input
          type="range"
          :min="config.aMin"
          :max="config.aMax"
          :step="config.aStep"
          v-model.number="a"
          class="slider-a"
        />
        <span class="slider-value-a">{{ a.toFixed(1) }}</span>
      </div>

      <!-- Order n -->
      <div class="flex items-center gap-2">
        <span class="slider-label-n">n</span>
        <input
          type="range"
          :min="0"
          :max="config.maxOrder"
          v-model.number="n"
          class="slider-n"
        />
        <span class="slider-value-n">{{ n }}</span>
      </div>
    </div>

    <VChart :option="option" autoresize class="flex-1" />
  </div>
</template>

<style scoped>
.control-bar {
  background: linear-gradient(135deg, rgba(96, 165, 250, 0.08), rgba(249, 115, 22, 0.08));
  border: 1px solid rgba(255, 255, 255, 0.06);
}

.slider-label-a {
  font-family: monospace;
  font-size: 0.8rem;
  font-weight: 700;
  color: #60a5fa;
  min-width: 1rem;
  text-align: center;
}

.slider-value-a {
  font-family: monospace;
  font-size: 0.8rem;
  font-weight: 700;
  color: #60a5fa;
  min-width: 2rem;
  text-align: center;
}

.slider-label-n {
  font-family: monospace;
  font-size: 0.8rem;
  font-weight: 700;
  color: #f97316;
  min-width: 1rem;
  text-align: center;
}

.slider-value-n {
  font-family: monospace;
  font-size: 0.8rem;
  font-weight: 700;
  color: #f97316;
  min-width: 1.5rem;
  text-align: center;
}

/* Slider a — blue theme */
.slider-a {
  -webkit-appearance: none;
  appearance: none;
  width: 90px;
  height: 5px;
  border-radius: 3px;
  background: linear-gradient(90deg, #1e40af, #60a5fa);
  outline: none;
  cursor: pointer;
}
.slider-a::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: #60a5fa;
  box-shadow: 0 0 6px rgba(96, 165, 250, 0.7);
  transition: box-shadow 0.2s;
}
.slider-a::-webkit-slider-thumb:hover {
  box-shadow: 0 0 12px rgba(96, 165, 250, 1);
}
.slider-a::-moz-range-thumb {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: #60a5fa;
  border: none;
  box-shadow: 0 0 6px rgba(96, 165, 250, 0.7);
}

/* Slider n — orange theme */
.slider-n {
  -webkit-appearance: none;
  appearance: none;
  width: 90px;
  height: 5px;
  border-radius: 3px;
  background: linear-gradient(90deg, #9a3412, #f97316);
  outline: none;
  cursor: pointer;
}
.slider-n::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: #f97316;
  box-shadow: 0 0 6px rgba(249, 115, 22, 0.7);
  transition: box-shadow 0.2s;
}
.slider-n::-webkit-slider-thumb:hover {
  box-shadow: 0 0 12px rgba(249, 115, 22, 1);
}
.slider-n::-moz-range-thumb {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: #f97316;
  border: none;
  box-shadow: 0 0 6px rgba(249, 115, 22, 0.7);
}
</style>
