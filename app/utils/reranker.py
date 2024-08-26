import numpy as np
import faiss
from typing import List, Dict
from app.models import FrameMetadataModel
from app.abstract_classes import AbstractIndexer


class Reranker:
    def __init__(self, indexer: AbstractIndexer):
        self.indexer = indexer

    async def rerank(self, prev_result: List[Dict], lst_pos_vote_idxs: List[int],
                     lst_neg_vote_idxs: List[int], k: int) -> List[FrameMetadataModel]:
        lst_vote_idxs = lst_pos_vote_idxs + lst_neg_vote_idxs
        len_pos = len(lst_pos_vote_idxs)

        result = {id: score for item in prev_result for id, score in zip(
            item['video_info']['lst_idxs'], item['video_info']['lst_scores'])}
        for key in lst_neg_vote_idxs:
            result.pop(key, None)

        id_selector = faiss.IDSelectorArray(
            np.array(list(result.keys())).astype('int64'))
        query_feats = self.indexer.clip_index.reconstruct_batch(
            np.array(lst_vote_idxs).astype('int64'))
        lst_scores, lst_idx_images = self.indexer.clip_index.search(query_feats, k=min(k, len(result)),
                                                                    params=faiss.SearchParametersIVF(sel=id_selector))

        for i, (scores, idx_images) in enumerate(zip(lst_scores, lst_idx_images)):
            for score, idx_image in zip(scores, idx_images):
                if 0 <= i < len_pos:
                    result[idx_image] += score
                else:
                    result[idx_image] -= score

        reranked_result = sorted(
            result.items(), key=lambda x: x[1], reverse=True)
        reranked_ids, reranked_scores = zip(*reranked_result)

        return self.get_frame_metadata(reranked_ids, reranked_scores)

    def get_frame_metadata(self, indices, scores) -> List[FrameMetadataModel]:
        results = []
        for idx, score in zip(indices, scores):
            frame = self.indexer.get_frame_by_index(int(idx))
            if frame:
                frame.score = score
                results.append(frame)
        return results
