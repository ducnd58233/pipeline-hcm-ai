from app.services.reranker.abstract_reranker import AbstractReranker
from app.models import FrameMetadataModel, ObjectQuery, Score
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer, util
import torch
from app.log import logger


class SentenceBertReranker(AbstractReranker):
    MODELS = {
        'default': "all-MiniLM-L6-v2",
        'powerful': "all-mpnet-base-v2",
        'multilingual': "paraphrase-multilingual-MiniLM-L12-v2",
        'efficient': "all-MiniLM-L12-v2"
    }

    def __init__(self, model_type: str = 'multilingual'):
        self.model_type = model_type
        try:
            self.model = SentenceTransformer(self.MODELS[model_type])
        except KeyError:
            logger.error(
                f"Unknown model type: {model_type}. Using default model.")
            self.model_type = 'default'
            self.model = SentenceTransformer(self.MODELS['default'])
        except Exception as e:
            logger.error(
                f"Error initializing SentenceTransformer model: {str(e)}")
            raise

    def switch_model(self, model_type: str):
        if model_type not in self.MODELS:
            logger.warning(
                f"Unknown model type: {model_type}. Keeping current model.")
            return
        if model_type != self.model_type:
            try:
                self.model_type = model_type
                self.model = SentenceTransformer(self.MODELS[model_type])
                logger.info(f"Switched to model: {model_type}")
            except Exception as e:
                logger.error(
                    f"Error switching to model {model_type}: {str(e)}")
                raise

    def rerank(self, merged_results: Dict[str, FrameMetadataModel], text_query: Optional[str], object_query: Optional[ObjectQuery]) -> List[FrameMetadataModel]:
        try:
            if text_query and len(text_query.split()) > 10:
                self.switch_model('powerful')

            object_query_str = ' '.join(
                object_query.parse_query()) if object_query else ''
            combined_query = f"{text_query or ''} {object_query_str}".strip()
            logger.info(f'SBertReranker query: {combined_query}')

            frame_texts = []
            for frame_id, frame in merged_results.items():
                frame_text = f"{frame.id} {frame.keyframe.video_path} {frame.keyframe.frame_path}"
                if frame.detection:
                    frame_text += ' ' + \
                        ' '.join([f"{obj}:{len(items)}" for obj,
                                 items in frame.detection.objects.items()])
                frame_texts.append(frame_text)

            # Encode query and frame texts
            query_embedding = self.model.encode(
                combined_query, convert_to_tensor=True)
            frame_embeddings = self.model.encode(
                frame_texts, convert_to_tensor=True)

            with torch.no_grad():
                cosine_scores = util.cos_sim(
                    query_embedding, frame_embeddings)[0]

            for (frame_id, frame), score in zip(merged_results.items(), cosine_scores):
                frame.score.details['rerank'] = float(score)
                frame.final_score = float(score)

            return sorted(merged_results.values(), key=lambda x: x.final_score, reverse=True)

        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            return list(merged_results.values())

    def get_model_info(self) -> Dict[str, str]:
        return {
            "current_model": self.model_type,
            "model_name": self.MODELS[self.model_type],
            "available_models": list(self.MODELS.keys())
        }
