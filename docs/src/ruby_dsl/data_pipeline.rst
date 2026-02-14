Data Pipeline
=============

``MLX::DSL::Data`` provides lazy, chainable enumerable transforms for trainer
input pipelines.

.. code-block:: ruby

   dataset = MLX::DSL::Data.from(records).batch(32)
   trainer.fit(dataset, epochs: 3)

Entry points
------------

- ``MLX::DSL::Data.from(source_or_block)``
- ``MLX::DSL::Data.pipeline(...)``

.. code-block:: ruby

   records = [{x: [1.0, 2.0], y: 0}, {x: [3.0, 4.0], y: 1}]

   from_enum = MLX::DSL::Data.from(records)
   from_block = MLX::DSL::Data.pipeline do
     Enumerator.new do |yielder|
       records.each { |row| yielder << row }
     end
   end

Transforms
----------

- ``map``
- ``filter``
- ``batch(size, drop_last:)``
- ``take(count)``
- ``repeat(times = nil)``
- ``shuffle(seed:, random:)``
- ``prefetch(size)``

.. code-block:: ruby

   batches = MLX::DSL::Data
     .from(records)
     .shuffle(seed: 7)
     .map { |row| [row[:x], row[:y]] }
     .batch(32, drop_last: false)
     .take(10)

Signature-aware map/filter
--------------------------

``map`` and ``filter`` support positional/keyword callable signatures with
``item``, ``index``, and ``pipeline`` context.

.. code-block:: ruby

   data = MLX::DSL::Data
     .from(records)
     .map { |item, index| preprocess(item, index: index) }
     .filter { |item, index:| index.even? }
     .batch(32)

See implementation:

- ``lib/mlx/dsl/data_pipeline.rb``
