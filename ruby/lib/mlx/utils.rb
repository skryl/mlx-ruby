# frozen_string_literal: true

module MLX
  module Utils
    module_function

    def tree_map(fn, tree, *rest, is_leaf: nil)
      if !is_leaf.nil? && is_leaf.call(tree)
        return fn.call(tree, *rest)
      end

      if tree.is_a?(Array)
        return tree.each_with_index.map do |child, i|
          tree_map(fn, child, *(rest.map { |r| r[i] }), is_leaf: is_leaf)
        end
      end

      if tree.is_a?(Hash)
        return tree.each_with_object({}) do |(k, child), out|
          out[k] = tree_map(fn, child, *(rest.map { |r| r[k] }), is_leaf: is_leaf)
        end
      end

      fn.call(tree, *rest)
    end

    def tree_map_with_path(fn, tree, *rest, is_leaf: nil, path: nil)
      if !is_leaf.nil? && is_leaf.call(tree)
        return fn.call(path, tree, *rest)
      end

      if tree.is_a?(Array)
        return tree.each_with_index.map do |child, i|
          next_path = path.nil? || path.empty? ? i.to_s : "#{path}.#{i}"
          tree_map_with_path(fn, child, *(rest.map { |r| r[i] }), is_leaf: is_leaf, path: next_path)
        end
      end

      if tree.is_a?(Hash)
        return tree.each_with_object({}) do |(k, child), out|
          next_path = path.nil? || path.empty? ? k.to_s : "#{path}.#{k}"
          out[k] = tree_map_with_path(
            fn,
            child,
            *(rest.map { |r| r[k] }),
            is_leaf: is_leaf,
            path: next_path
          )
        end
      end

      fn.call(path, tree, *rest)
    end

    def tree_flatten(tree, prefix: "", is_leaf: nil, destination: nil)
      destination ||= []
      unless destination.is_a?(Array) || destination.is_a?(Hash)
        raise ArgumentError, "destination must be an Array, Hash, or nil"
      end

      add = if destination.is_a?(Array)
        ->(k, v) { destination << [k, v] }
      else
        ->(k, v) { destination[k] = v }
      end

      recurse = lambda do |node, current|
        if !is_leaf.nil? && is_leaf.call(node)
          add.call(current, node)
          next
        end

        if node.is_a?(Array)
          node.each_with_index do |child, i|
            key = current.empty? ? i.to_s : "#{current}.#{i}"
            recurse.call(child, key)
          end
        elsif node.is_a?(Hash)
          node.each do |k, child|
            key = current.empty? ? k.to_s : "#{current}.#{k}"
            recurse.call(child, key)
          end
        else
          add.call(current, node)
        end
      end

      recurse.call(tree, prefix.to_s.sub(/\A\./, ""))
      destination
    end

    def tree_unflatten(tree)
      items = if tree.is_a?(Hash)
        tree.to_a
      elsif tree.is_a?(Array)
        tree
      else
        raise ArgumentError, "tree must be an Array of pairs or Hash"
      end

      if items.length == 1 && items[0][0].to_s.empty?
        return items[0][1]
      end

      children = Hash.new { |h, k| h[k] = [] }
      items.each do |key, value|
        parts = key.to_s.split(".", 2)
        current = parts[0]
        rest = parts.length == 2 ? parts[1] : ""
        children[current] << [rest, value]
      end

      numeric = children.keys.all? { |k| /\A\d+\z/.match?(k) }
      if numeric
        out = []
        children.keys.map(&:to_i).sort.each do |idx|
          key = idx.to_s
          out[idx] = tree_unflatten(children[key])
        end
        return out
      end

      children.each_with_object({}) do |(k, pairs), out|
        out[k] = tree_unflatten(pairs)
      end
    end

    def tree_reduce(fn, tree, initializer = nil, is_leaf: nil)
      if !is_leaf.nil? && is_leaf.call(tree)
        return initializer.nil? ? tree : fn.call(initializer, tree)
      end

      acc = initializer
      if tree.is_a?(Array)
        tree.each { |item| acc = tree_reduce(fn, item, acc, is_leaf: is_leaf) }
      elsif tree.is_a?(Hash)
        tree.each_value { |item| acc = tree_reduce(fn, item, acc, is_leaf: is_leaf) }
      else
        return acc.nil? ? tree : fn.call(acc, tree)
      end
      acc
    end

    def tree_merge(tree_a, tree_b, merge_fn: nil)
      tree_a = nil if (tree_a.is_a?(Array) || tree_a.is_a?(Hash)) && tree_a.empty?
      tree_b = nil if (tree_b.is_a?(Array) || tree_b.is_a?(Hash)) && tree_b.empty?

      return tree_b if tree_a.nil? && !tree_b.nil?
      return tree_a if !tree_a.nil? && tree_b.nil?

      if tree_a.is_a?(Array) && tree_b.is_a?(Array)
        max = [tree_a.length, tree_b.length].max
        return Array.new(max) do |i|
          tree_merge(tree_a[i], tree_b[i], merge_fn: merge_fn)
        end
      end

      if tree_a.is_a?(Hash) && tree_b.is_a?(Hash)
        keys = tree_a.keys | tree_b.keys
        return keys.each_with_object({}) do |k, out|
          out[k] = tree_merge(tree_a[k], tree_b[k], merge_fn: merge_fn)
        end
      end

      if merge_fn.nil?
        raise ArgumentError, "merge_fn required when merging leaf values"
      end
      merge_fn.call(tree_a, tree_b)
    end
  end
end
