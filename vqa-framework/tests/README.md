
`--runhuman` flag to run tests that require a person. Mark tests with `@pytest.mark.human`

`--runslow` flag to run slower tests. Mark tests with `@pytest.mark.slow`

`--runglobaldataset` flag to run tests that use global dataset locations
(i.e., the full dataset). Probably slow, and risks damaging your already downloaded
datasets. Also risks overwriting something if the test hasn't been updated & is 
still using old code. Mark tests with `@pytest.mark.globaldataset`
