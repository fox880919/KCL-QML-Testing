from sklearn.decomposition import PCA

class MyPCA:

    def implementPCA(self, x_tr, x_test, pca_n_components):

        pca = PCA(n_components = pca_n_components)
        xs_tr = pca.fit_transform(x_tr)
        xs_test = pca.transform(x_test)

        return xs_tr, xs_test