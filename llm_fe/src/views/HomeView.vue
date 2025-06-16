<template>
  <div class="p-4">
    <div class="flex flex-col items-center mb-10">
      <div class="flex flex-row gap-20 justify-center">
        <div class="flex items-center flex-col">
          <h2 class="text-3xl font-semibold mb-4">Select Authors</h2>
          <div class="grid grid-cols-2 gap-2 mb-4">
            <div v-for="author in allAuthors" :key="author" class="flex items-center gap-2 text-l">
              <input
                :id="author"
                class="w-5 h-5"
                type="checkbox"
                :value="author"
                v-model="selectedAuthors"
              />
              <label :for="author">{{ authorMap[author] }}</label>
            </div>
          </div>
        </div>
        <div>
          <h2 class="text-3xl font-semibold mb-4">Select Model</h2>
          <div class="grid gap-2 mb-4">
            <div v-for="model in allModels" :key="model" class="flex items-center gap-2 text-l">
              <input
                :id="model"
                class="w-5 h-5"
                type="radio"
                :value="model"
                v-model="selectedModel"
              />
              <label :for="model">{{ modelMap[model] }}</label>
            </div>
          </div>
        </div>
      </div>
      <div>
        <button @click="fetchTSNE" class="px-4 py-2 bg-blue-500 text-white rounded">
          Fetch Authors Data
        </button>
      </div>
    </div>

    <div v-if="filteredAuthorsLoading" class="flex justify-center items-center h-40">
      <div
        class="animate-spin rounded-full h-10 w-10 border-t-4 border-blue-500 border-solid"
      ></div>
    </div>
    <div v-else-if="filteredAuthorsPlotData.length" class="mt-6 w-full h-[70vh]">
      <VuePlotly
        :data="filteredAuthorsPlotData"
        :layout="filteredAuthorsPlotLayout"
        :config="config"
        :use-resize-handler="true"
        style="width: 100%; height: 100%"
      />
    </div>

    <div class="flex flex-col items-center mb-10">
      <h2 class="text-3xl font-semibold mb-4">Per Book Embeddings</h2>
      <button @click="fetchTsneBooks" class="px-4 py-2 bg-blue-500 text-white rounded">
        Fetch Books Data
      </button>
    </div>

    <div v-if="tsneBooksLoading" class="flex justify-center items-center h-40">
      <div
        class="animate-spin rounded-full h-10 w-10 border-t-4 border-blue-500 border-solid"
      ></div>
    </div>
    <div v-else-if="tsneBooksPlotData.length" class="mt-6 w-full h-[70vh]">
      <VuePlotly
        :data="tsneBooksPlotData"
        :layout="tsneBooksPlotLayout"
        :config="config"
        :use-resize-handler="true"
        style="width: 100%; height: 100%"
      />
    </div>

    <div class="flex flex-col items-center mb-4">
      <h2 class="text-3xl font-semibold mb-4">Averages for Authors</h2>
      <button @click="fetchTsneAuthorAverage" class="px-4 py-2 bg-blue-500 text-white rounded">
        Fetch Average Author Data
      </button>
    </div>

    <div v-if="tsneAuthorAverageLoading" class="flex justify-center items-center h-40">
      <div
        class="animate-spin rounded-full h-10 w-10 border-t-4 border-blue-500 border-solid"
      ></div>
    </div>
    <div v-else-if="tsneAuthorAveragePlotData.length" class="mt-6 w-full h-[70vh]">
      <VuePlotly
        :data="tsneAuthorAveragePlotData"
        :layout="tsneAuthorAverageLayout"
        :config="config"
        :use-resize-handler="true"
        style="width: 100%; height: 100%"
      />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'
import { VuePlotly } from 'vue3-plotly'

const authorMap = {
  fredro_aleksander: 'Aleksander Fredro',
  kochanowski_jan: 'Jan Kochanowski',
  krasicki_ignacy: 'Ignacy Krasicki',
  mickiewicz_adam: 'Adam Mickiewicz',
  prus_bolesław: 'Bolesław Prus',
  sienkiewicz_henryk: 'Henryk Sienkiewicz',
  skarga_piotr: 'Piotr Skarga',
  słowacki_juliusz: 'Juliusz Słowacki',
  baczynski_krzysztof_kamil: 'K.K. Baczyński',
  konopnicka_maria: 'Maria Konopnicka',
}
const modelMap = {
  mistral: 'Mistral',
  'mwiewior/bielik': 'Bielik',
  'antoniprzybylik/llama-pllum:8b': 'Pllum',
}
const allModels = Object.keys(modelMap)
const selectedModel = ref(allModels[2])

const fileMap = {
  'faraon-tom-drugi.txt': 'Faraon (Tom II)',
  'krzyzacy-tom-drugi.txt': 'Krzyżacy (Tom II)',
  'ogniem-i-mieczem-tom-drugi.txt': 'Ogniem i Mieczem (Tom II)',
  'lalka-tom-pierwszy.txt': 'Lalka (Tom I)',
  'lalka-tom-drugi.txt': 'Lalka (Tom II)',
  'skarga-kazania-sejmowe.txt': 'Kazania Sejmowe',
  'quo-vadis.txt': 'Quo Vadis',
  'krzyzacy-tom-pierwszy.txt': 'Krzyżacy (Tom I)',
  'faraon-tom-pierwszy.txt': 'Faraon (Tom I)',
  'faraon-tom-trzeci.txt': 'Faraon (Tom III)',
  'ogniem-i-mieczem-tom-pierwszy.txt': 'Ogniem i Mieczem (Tom I)',
  'zemsta.txt': 'Zemsta',
  'balladyna.txt': 'Balladyna',
  'fredro-sluby-panienskie.txt': 'Śluby Panieńskie',
  'pan-tadeusz.txt': 'Pan Tadeusz',
  'pan-wolodyjowski.txt': 'Pan Wołodyjowski',
  'w-pustyni-i-w-puszczy.txt': 'W pustyni i w puszczy',
  'kordian.txt': 'Kordian',
  'dziady-dziady-poema-dziady-czesc-iii.txt': 'Dziady (Część III)',
  'piesn-swietojanska-o-sobotce.txt': 'Pieśń Świętojańska o Sobótce',
  'latarnik.txt': 'Latarnik',
  'monachomachia.txt': 'Monachomachia',
  'odprawa-poslow-greckich.txt': 'Odprawa posłów greckich',
  'kamizelka.txt': 'Kamizelka',
  'kochanowski-song-xxv.txt': 'Kochanowski - Pieśń XXV',
  'baczynski-ballada-o-rzece.txt': 'Ballada o Rzece',
  'o-janku-wedrowniczku.txt': 'O Janku Wędrowniczku',
  'baczynski-pokolenie-do-palcow-przymarzly-struny.txt': 'Pokolenie (Do palców przymarzły struny)',
}

const allAuthors = Object.keys(authorMap)
const selectedAuthors = ref([])

const filteredAuthorsPlotData = ref([])
const filteredAuthorsPlotLayout = ref({
  title: {
    text: 't-SNE Visualization by Selected Authors',
    font: { color: '#ffffff' }, // Title color
  },
  xaxis: {
    title: 't-SNE 1',
    color: '#ffffff', // Axis tick & label color
    titlefont: { color: '#ffffff' },
  },
  yaxis: {
    title: 't-SNE 2',
    color: '#ffffff',
    titlefont: { color: '#ffffff' },
  },
  legend: {
    font: { color: '#ffffff' },
  },
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  autosize: true,
})

const tsneBooksPlotData = ref([])
const tsneBooksPlotLayout = ref({
  title: {
    text: 't-SNE Visualization by Books',
    font: { color: '#ffffff' }, // Title color
  },
  xaxis: {
    title: 't-SNE 1',
    color: '#ffffff', // Axis tick & label color
    titlefont: { color: '#ffffff' },
  },
  yaxis: {
    title: 't-SNE 2',
    color: '#ffffff',
    titlefont: { color: '#ffffff' },
  },
  legend: {
    font: { color: '#ffffff' },
  },
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  autosize: true,
})

const tsneAuthorAveragePlotData = ref([])
const tsneAuthorAverageLayout = ref({
  title: {
    text: 't-SNE Visualization by Author Average',
    font: { color: '#ffffff' }, // Title color
  },
  xaxis: {
    title: 't-SNE 1',
    color: '#ffffff', // Axis tick & label color
    titlefont: { color: '#ffffff' },
  },
  yaxis: {
    title: 't-SNE 2',
    color: '#ffffff',
    titlefont: { color: '#ffffff' },
  },
  legend: {
    font: { color: '#ffffff' },
  },
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  autosize: true,
})

const config = {
  responsive: true,
}
const filteredAuthorsLoading = ref(false)
const tsneBooksLoading = ref(false)
const tsneAuthorAverageLoading = ref(false)

async function fetchTSNE() {
  filteredAuthorsLoading.value = true
  try {
    const params = {
      model_name: selectedModel.value,
    }

    if (selectedAuthors.value.length > 0) {
      params.authors = selectedAuthors.value.join(',')
    }

    const res = await axios.get('http://localhost:8000/api/tsne', { params })

    console.log(res.data)

    // group points by author for Plotly
    const grouped = {}
    res.data.forEach(({ x, y, author }) => {
      if (!grouped[author]) {
        grouped[author] = {
          x: [],
          y: [],
          mode: 'markers',
          type: 'scatter',
          name: authorMap[author] || author,
          marker: {
            size: 5,
          },
        }
      }
      grouped[author].x.push(x)
      grouped[author].y.push(y)
    })

    filteredAuthorsPlotData.value = Object.values(grouped)
  } catch (err) {
    console.error('Failed to load t-SNE data', err)
  } finally {
    filteredAuthorsLoading.value = false
  }
}

async function fetchTsneBooks() {
  tsneBooksLoading.value = true
  try {
    const params = {
      model_name: selectedModel.value,
    }

    const res = await axios.get('http://localhost:8000/api/tsne-books', { params })

    console.log(res.data)

    const grouped = {}
    res.data.forEach(({ x, y, label }) => {
      if (!grouped[label]) {
        grouped[label] = {
          x: [],
          y: [],
          mode: 'markers',
          type: 'scatter',
          name: fileMap[label] || label,
          marker: {
            size: 5,
          },
        }
      }
      grouped[label].x.push(x)
      grouped[label].y.push(y)
    })

    tsneBooksPlotData.value = Object.values(grouped)
  } catch (err) {
    console.error('Failed to load t-SNE data', err)
  } finally {
    tsneBooksLoading.value = false
  }
}

async function fetchTsneAuthorAverage() {
  tsneAuthorAverageLoading.value = true
  try {
    const params = {
      model_name: selectedModel.value,
    }

    const res = await axios.get('http://localhost:8000/api/author-average-tsne', { params })

    console.log(res.data)

    const grouped = {}
    res.data.forEach(({ x, y, author }) => {
      if (!grouped[author]) {
        grouped[author] = {
          x: [],
          y: [],
          mode: 'markers',
          type: 'scatter',
          name: authorMap[author] || author,
          marker: {
            size: 15,
          },
        }
      }
      grouped[author].x.push(x)
      grouped[author].y.push(y)
    })

    tsneAuthorAveragePlotData.value = Object.values(grouped)
  } catch (err) {
    console.error('Failed to load t-SNE data', err)
  } finally {
    tsneAuthorAverageLoading.value = false
  }
}
</script>
