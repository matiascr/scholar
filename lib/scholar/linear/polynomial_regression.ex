defmodule Scholar.Linear.PolynomialRegression do
  @moduledoc """
  Least squares polynomial regression.
  """

  import Nx.Defn

  @derive {Nx.Container, containers: [:coefficients, :intercept, :degree]}
  defstruct [:coefficients, :intercept, :degree]

  opts = [
    sample_weights: [
      type: {:list, {:custom, Scholar.Options, :positive_number, []}},
      doc: """
      The weights for each observation. If not provided,
      all observations are assigned equal weight.
      """
    ],
    degree: [
      type: :pos_integer,
      default: 2,
      doc: """
      The degree of the feature matrix to return. Must be a >1 integer. 1
      returns the input matrix.
      """
    ],
    fit_intercept?: [
      type: :boolean,
      default: true,
      doc: """
      If set to `true`, a model will fit the intercept. Otherwise,
      the intercept is set to `0.0`. The intercept is an independent term
      in a linear model. Specifically, it is the expected mean value
      of targets for a zero-vector on input.
      """
    ]
  ]

  transform_opts = Keyword.take(opts, [:degree, :fit_intercept])

  @opts_schema NimbleOptions.new!(opts)
  @transform_opts_schema NimbleOptions.new!(transform_opts)

  @doc """
  Fits a polynomial regression model for sample inputs `a` and
  sample targets `b`.

  ## Options

  #{NimbleOptions.docs(@opts_schema)}

  ## Return Values

  The function returns a struct with the following parameters:

    * `:coefficients` - Estimated coefficients for the polynomial regression problem.

    * `:intercept` - Independent term in the polynomial model.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([4.0, 3.0, -1.0])
      iex> Scholar.Linear.PolynomialRegression.fit(x, y, degree: 1)
      %Scholar.Linear.PolynomialRegression{
        coefficients: #Nx.Tensor<
          f32[2]
          [-0.4972473084926605, -0.70103919506073]
        >, intercept: #Nx.Tensor<
          f32
          5.896470069885254
        >
      }

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([4.0, 3.0, -1.0])
      iex> model = Scholar.Linear.PolynomialRegression.fit(x, y, degree: 2)
      %Scholar.Linear.PolynomialRegression{
        coefficients: #Nx.Tensor<
          f32[5]
          [-0.021396497264504433, -0.004854593891650438, -0.08849877119064331, -0.062211357057094574, -0.04369127377867699]
        >, intercept: #Nx.Tensor<
          f32
          4.418517112731934
        >
      }
  """
  deftransform fit(a, b, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    a_transform = transform(a, opts |> Keyword.put(:fit_intercept?, false))

    %{
      Scholar.Linear.LinearRegression.fit(a_transform, b, Keyword.delete(opts, :degree))
      | __struct__: Scholar.Linear.PolynomialRegression
    }
    |> Map.merge(%{degree: opts[:degree]})
  end

  @doc """
  Makes predictions with the given `model` on input `x`.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 2.0], [4.0, 7.0]])
      iex> y = Nx.tensor([4.0, 3.0, -1.0])
      iex> model = Scholar.Linear.PolynomialRegression.fit(x, y, degree: 2)
      iex> Scholar.Linear.PolynomialRegression.predict(model, Nx.tensor([[2.0, 1.0]]))
      #Nx.Tensor<
        f32[1]
        [3.84888117]
      >
  """
  deftransform predict(model, x) do
    Scholar.Linear.LinearRegression.predict(
      %{model | __struct__: Scholar.Linear.LinearRegression},
      transform(x, degree: model.degree, fit_intercept?: false)
    )
  end

  @doc """
  Computes the feature matrix for polynomial regression.

  #{NimbleOptions.docs(@transform_opts_schema)}

  ## Examples

      iex> x = Nx.tensor([[2]])
      iex> Scholar.Linear.PolynomialRegression.transform(x, degree: 0)
      ** (NimbleOptions.ValidationError) invalid value for :degree option: expected positive integer, got: 0

      iex> x = Nx.tensor([[2]])
      iex> Scholar.Linear.PolynomialRegression.transform(x, degree: 5, fit_intercept?: false)
      #Nx.Tensor<
        s64[1][5]
        [
          [2, 4, 8, 16, 32]
        ]
      >

      iex> x = Nx.tensor([[2, 3]])
      iex> Scholar.Linear.PolynomialRegression.transform(x)
      #Nx.Tensor<
        s64[1][6]
        [
          [1, 2, 3, 4, 6, 9]
        ]
      >

      iex> x = Nx.iota({3, 2})
      iex> Scholar.Linear.PolynomialRegression.transform(x, fit_intercept?: false)
      #Nx.Tensor<
        s64[3][5]
        [
          [0, 1, 0, 0, 1],
          [2, 3, 4, 6, 9],
          [4, 5, 16, 20, 25]
        ]
      >
  """
  deftransform transform(x, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    transform_n(x, initial_xp(x, opts), opts)
  end

  defn transform_n(x, xp, opts) do
    indices = Nx.iota({Nx.shape(x) |> elem(1)})

    {_, _, _, _, xp} =
      while {d = 1, indices, start = 0, x, xp}, d < opts[:degree] do
        {_, indices, start, _, xp} = compute_features(indices, start, x, xp)

        {d + 1, indices, start, x, xp}
      end

    xp
  end

  defn compute_features(indices, start, x, xp) do
    {n_samples, n_features} = Nx.shape(x)
    l = Nx.size(indices)

    while {i = 0, indices, start, x, xp}, i < l do
      factor_col = Nx.transpose(x)[i] |> Nx.reshape({n_samples, :auto})

      previous_deg_cols =
        Nx.transpose(xp)[start]
        |> Nx.reshape({n_samples, :auto})

      {_, new_size} = Nx.shape(previous_deg_cols)

      xp = Nx.put_slice(xp, [0, n_features + start], factor_col * previous_deg_cols)

      {i + 1, indices, start + new_size, x, xp}
    end
  end

  deftransform initial_xp(x, opts) do
    {n_samples, _n_features} = Nx.shape(x)

    :nan
    |> Nx.broadcast({n_samples, Scholar.Metrics.comb(x, opts[:degree])})
    |> Nx.put_slice([0, 0], x)
  end
end
