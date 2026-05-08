<script setup>
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { ScatterChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([ScatterChart, GridComponent, TooltipComponent, CanvasRenderer])

const props = defineProps({
  vectors: {
    type: Array,
    default: () => [
      { x: 3, y: 1, label: 'v', color: '#60a5fa' },
      { x: 1, y: 2, label: 'u', color: '#34d399' },
    ],
  },
  showGrid: { type: Boolean, default: true },
})

const option = {
  animation: false,
  grid: { left: 50, right: 30, top: 20, bottom: 40 },
  xAxis: {
    type: 'value',
    min: -0.5,
    max: 5,
    splitLine: { lineStyle: { color: '#333' } },
    axisLine: { lineStyle: { color: '#666' } },
    axisLabel: { color: '#aaa' },
    name: 'x₁',
    nameTextStyle: { color: '#aaa' },
  },
  yAxis: {
    type: 'value',
    min: -0.5,
    max: 4,
    splitLine: { lineStyle: { color: '#333' } },
    axisLine: { lineStyle: { color: '#666' } },
    axisLabel: { color: '#aaa' },
    name: 'x₂',
    nameTextStyle: { color: '#aaa' },
  },
  series: props.vectors.map(v => ({
    type: 'scatter',
    data: [[v.x, v.y]],
    symbolSize: 8,
    itemStyle: { color: v.color },
    label: {
      show: true,
      formatter: `${v.label} = (${v.x}, ${v.y})`,
      position: 'right',
      color: v.color,
      fontSize: 13,
    },
    markLine: {
      silent: true,
      symbol: ['none', 'arrow'],
      symbolSize: 10,
      lineStyle: { color: v.color, width: 2 },
      data: [{ coord: [0, 0], coord2: [v.x, v.y] }],
    },
  })),
}
</script>

<template>
  <VChart :option="option" autoresize style="width: 100%; height: 100%;" />
</template>
