import type { NextPage } from 'next'
import Head from 'next/head'
import SoundInput from '../components/SoundInput'

const Home: NextPage = () => {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center py-2">
      <Head>
        <title>Minor Project</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="flex w-full flex-1 flex-col items-center justify-center px-20 text-center">
        <SoundInput />
      </main> 
    </div>
  )
}

export default Home
