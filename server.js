import express from 'express';
import dotenv from 'dotenv';
import { HfInference } from '@huggingface/inference';

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());
app.use(express.static('.'));

const hf = new HfInference(process.env.HF_ACCESS_TOKEN);

app.post('/api/generate', async (req, res) => {
    try {
        const { prompt } = req.body;
        if (!prompt) {
            return res.status(400).json({ error: 'Prompt is required' });
        }

        // Using FLUX.1-schnell (currently active on free Hugging Face serverless)
        const imageBlob = await hf.textToImage({
            model: 'black-forest-labs/FLUX.1-schnell',
            inputs: prompt,
        });

        const buffer = Buffer.from(await imageBlob.arrayBuffer());
        const imageUrl = `data:image/jpeg;base64,${buffer.toString('base64')}`;

        res.json({ imageUrl });
    } catch (error) {
        console.error('Generation Error:', error);
        res.status(500).json({ error: error.message || 'Failed to generate image' });
    }
});

app.listen(port, () => {
    console.log(`Forge AI server running at http://localhost:${port}`);
});