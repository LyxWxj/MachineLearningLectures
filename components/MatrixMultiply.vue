<template>
  <div class="flex flex-col items-center gap-4">
    <div class="text-sm text-gray-400">
      Click a cell in C to see the calculation
    </div>

    <div class="flex items-center gap-6 flex-wrap justify-center">
      <!-- Matrix A -->
      <div class="flex flex-col items-center">
        <div class="text-lg font-bold mb-2">A ({{ rowsA }} × {{ colsA }})</div>
        <svg :width="colsA * cellSize + 20" :height="rowsA * cellSize + 20">
          <template v-for="(row, i) in matrixA" :key="'a-' + i">
            <template v-for="(val, j) in row" :key="'a-' + i + '-' + j">
              <rect
                :x="j * cellSize + 10"
                :y="i * cellSize + 10"
                :width="cellSize"
                :height="cellSize"
                :fill="selectedRow === i ? '#0891b2' : '#1e293b'"
                :stroke="selectedRow === i ? '#22d3ee' : '#475569'"
                stroke-width="1.5"
                :opacity="0.7"
              />
              <text
                :x="j * cellSize + 10 + cellSize / 2"
                :y="i * cellSize + 10 + cellSize / 2"
                text-anchor="middle"
                dominant-baseline="central"
                :fill="selectedRow === i ? '#4ade80' : '#94a3b8'"
                :font-size="14"
                :font-weight="selectedRow === i ? 'bold' : 'normal'"
              >
                {{ val }}
              </text>
            </template>
          </template>
        </svg>
      </div>

      <!-- Multiplication sign -->
      <div class="text-3xl font-bold">×</div>

      <!-- Matrix B -->
      <div class="flex flex-col items-center">
        <div class="text-lg font-bold mb-2">B ({{ rowsB }} × {{ colsB }})</div>
        <svg :width="colsB * cellSize + 20" :height="rowsB * cellSize + 20">
          <template v-for="(row, i) in matrixB" :key="'b-' + i">
            <template v-for="(val, j) in row" :key="'b-' + i + '-' + j">
              <rect
                :x="j * cellSize + 10"
                :y="i * cellSize + 10"
                :width="cellSize"
                :height="cellSize"
                :fill="selectedCol === j ? '#0891b2' : '#1e293b'"
                :stroke="selectedCol === j ? '#22d3ee' : '#475569'"
                stroke-width="1.5"
                :opacity="0.7"
              />
              <text
                :x="j * cellSize + 10 + cellSize / 2"
                :y="i * cellSize + 10 + cellSize / 2"
                text-anchor="middle"
                dominant-baseline="central"
                :fill="selectedCol === j ? '#4ade80' : '#94a3b8'"
                :font-size="14"
                :font-weight="selectedCol === j ? 'bold' : 'normal'"
              >
                {{ val }}
              </text>
            </template>
          </template>
        </svg>
      </div>

      <!-- Equals sign -->
      <div class="text-3xl font-bold">=</div>

      <!-- Matrix C -->
      <div class="flex flex-col items-center">
        <div class="text-lg font-bold mb-2">C ({{ rowsA }} × {{ colsB }})</div>
        <svg :width="colsB * cellSize + 20" :height="rowsA * cellSize + 20">
          <template v-for="(row, i) in matrixC" :key="'c-' + i">
            <template v-for="(val, j) in row" :key="'c-' + i + '-' + j">
              <rect
                :x="j * cellSize + 10"
                :y="i * cellSize + 10"
                :width="cellSize"
                :height="cellSize"
                :fill="selectedRow === i && selectedCol === j ? '#0891b2' : '#1e293b'"
                :stroke="selectedRow === i && selectedCol === j ? '#22d3ee' : '#475569'"
                stroke-width="1.5"
                :opacity="0.7"
                class="cursor-pointer"
                @click="selectCell(i, j)"
              />
              <text
                :x="j * cellSize + 10 + cellSize / 2"
                :y="i * cellSize + 10 + cellSize / 2"
                text-anchor="middle"
                dominant-baseline="central"
                :fill="selectedRow === i && selectedCol === j ? '#4ade80' : '#94a3b8'"
                :font-size="14"
                :font-weight="selectedRow === i && selectedCol === j ? 'bold' : 'normal'"
                class="cursor-pointer pointer-events-none"
              >
                {{ val }}
              </text>
            </template>
          </template>
        </svg>
      </div>
    </div>

    <!-- Calculation details -->
    <div v-if="selectedRow !== null && selectedCol !== null" class="mt-4 p-4 bg-gray-800 rounded-lg text-center">
      <div class="text-lg mb-2">
        <span class="text-cyan-400">A[{{ selectedRow }}, :]</span>
        <span class="mx-2">·</span>
        <span class="text-cyan-400">B[:, {{ selectedCol }}]</span>
        <span class="mx-2">=</span>
        <span class="text-cyan-400">C[{{ selectedRow }}, {{ selectedCol }}]</span>
      </div>
      <div class="text-gray-300">
        {{ calculationText }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

const props = defineProps({
  matrixA: {
    type: Array as () => number[][],
    default: () => [
      [1, 2, 1],
      [2, 1, 2]
    ]
  },
  matrixB: {
    type: Array as () => number[][],
    default: () => [
      [1, 0, 1],
      [0, 1, 0],
      [1, 0, 0]
    ]
  },
  cellSize: { type: Number, default: 50 }
})

const selectedRow = ref<number | null>(null)
const selectedCol = ref<number | null>(null)

const rowsA = computed(() => props.matrixA.length)
const colsA = computed(() => props.matrixA[0].length)
const rowsB = computed(() => props.matrixB.length)
const colsB = computed(() => props.matrixB[0].length)

const matrixC = computed(() => {
  const result: number[][] = []
  for (let i = 0; i < rowsA.value; i++) {
    const row: number[] = []
    for (let j = 0; j < colsB.value; j++) {
      let sum = 0
      for (let k = 0; k < colsA.value; k++) {
        sum += props.matrixA[i][k] * props.matrixB[k][j]
      }
      row.push(sum)
    }
    result.push(row)
  }
  return result
})

const calculationText = computed(() => {
  if (selectedRow.value === null || selectedCol.value === null) return ''

  const i = selectedRow.value
  const j = selectedCol.value
  const terms: string[] = []

  for (let k = 0; k < colsA.value; k++) {
    terms.push(`${props.matrixA[i][k]} × ${props.matrixB[k][j]}`)
  }

  return terms.join(' + ') + ' = ' + matrixC.value[i][j]
})

function selectCell(i: number, j: number) {
  if (selectedRow.value === i && selectedCol.value === j) {
    selectedRow.value = null
    selectedCol.value = null
  } else {
    selectedRow.value = i
    selectedCol.value = j
  }
}
</script>
