from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_mongodb.retrievers.hybrid_search import MongoDBAtlasHybridSearchRetriever
from langchain_mongodb.pipelines import (
    combine_pipelines,
    final_hybrid_stage,
    reciprocal_rank_stage,
    text_search_stage,
    vector_search_stage,
)
from langchain_mongodb.utils import make_serializable

class CustomMongoDBAtlasHybridSearchRetriever(MongoDBAtlasHybridSearchRetriever):
    """
    Custom retriever that overrides _aget_relevant_documents to properly await
    the async aggregate call.
    """
    
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Build query vector synchronously.
        query_vector = self.vectorstore._embedding.embed_query(query)
        scores_fields = ["vector_score", "fulltext_score"]
        pipeline: List[Any] = []

        # Build vector search pipeline.
        vector_pipeline = [
            vector_search_stage(
                query_vector=query_vector,
                search_field=self.vectorstore._embedding_key,
                index_name=self.vectorstore._index_name,
                top_k=self.top_k,
                filter=self.pre_filter,
                oversampling_factor=self.oversampling_factor,
            )
        ]
        vector_pipeline += reciprocal_rank_stage("vector_score", self.vector_penalty)
        combine_pipelines(pipeline, vector_pipeline, self.collection.name)

        # Build text search pipeline.
        text_pipeline = text_search_stage(
            query=query,
            search_field=self.vectorstore._text_key,
            index_name=self.search_index_name,
            limit=self.top_k,
            filter=self.pre_filter,
        )
        text_pipeline.extend(reciprocal_rank_stage("fulltext_score", self.fulltext_penalty))
        combine_pipelines(pipeline, text_pipeline, self.collection.name)

        # Sum and sort stage.
        pipeline.extend(final_hybrid_stage(scores_fields=scores_fields, limit=self.top_k))

        # Remove embeddings unless requested.
        if not self.show_embeddings:
            pipeline.append({"$project": {self.vectorstore._embedding_key: 0}})
        if self.post_filter is not None:
            pipeline.extend(self.post_filter)

        # IMPORTANT: Await the aggregate call to get an async iterator.
        cursor = await self.collection.aggregate(pipeline)
        docs = []
        async for res in cursor:
            text = res.pop(self.vectorstore._text_key)
            make_serializable(res)
            docs.append(Document(page_content=text, metadata=res))
        return docs
