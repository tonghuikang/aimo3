import kaggle_evaluation.core.templates

import aimo_3_gateway


class AIMO3InferenceServer(kaggle_evaluation.core.templates.InferenceServer):
    def _get_gateway_for_test(self, data_paths=None, *args, **kwargs):
        return aimo_3_gateway.AIMO3Gateway(data_paths)
