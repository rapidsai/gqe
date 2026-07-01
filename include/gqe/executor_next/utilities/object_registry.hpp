/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "gqe/utility/error.hpp"

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace gqe::executor_next {

namespace detail {

/**
 * @brief A registered object along with the mutex that protects it.
 * @remark Not intended for use outside of Registry and registered_object_guard.
 * @tparam ObjectT The type of object that is registered.
 */
template <typename ObjectT>
class registered_object {
  static_assert(std::is_move_constructible_v<ObjectT>, "ObjectT must be move constructible.");

 public:
  /**
   * @brief Construct a new registered object and takes ownership of \a object.
   * @param[in] object The object being stored.
   */
  explicit registered_object(ObjectT object) : _object(std::move(object)) {}

  /**
   * @brief Get's a constant reference to the stored object.
   * @return ObjectT const&
   */
  [[nodiscard]] ObjectT const& get() const noexcept { return _object; }

  /**
   * @brief Get's a reference to the stored object.
   * @return ObjectT&
   */
  [[nodiscard]] ObjectT& get() noexcept { return _object; }

  /**
   * @brief Locks the internal mutex for shared access.
   */
  void lock_shared() { _mutex.lock_shared(); }

  /**
   * @brief Locks the internal mutex for shared access.
   */
  void unlock_shared() { _mutex.unlock_shared(); }

  /**
   * @brief Locks the internal mutex for unregistration.
   */
  void lock() { _mutex.lock(); }

  /**
   * @brief Unlocks the internal mutex for unregistration.
   */
  void unlock() { _mutex.unlock(); }

 private:
  ObjectT _object;           //< The registered object.
  std::shared_mutex _mutex;  //< The mutex to protect the registered object.
};

}  // namespace detail

/**
 * @brief Guards a registered ObjectT from concurrent access.
 * @tparam ObjectT The type of object being guarded.
 */
template <typename ObjectT>
class registered_object_guard {
 public:
  using registry_object_type = detail::registered_object<ObjectT>;

  /**
   * @brief Construct a new registered_object_guard from a registered_object
   *    and lock the registered_object.
   * @param[in] registered_object The registered_object being guarded.
   */
  explicit registered_object_guard(std::shared_ptr<registry_object_type> registered_object)
    : _registered_object(registered_object)
  {
    if (_registered_object) { _registered_object->lock_shared(); }
  }

  /**
   * @brief Destroy the registered_object_guard, unlocking the registered_object.
   */
  ~registered_object_guard() noexcept
  {
    if (_registered_object) { _registered_object->unlock_shared(); }
  }

  registered_object_guard(registered_object_guard const&)            = delete;
  registered_object_guard& operator=(registered_object_guard const&) = delete;
  registered_object_guard& operator=(registered_object_guard&&)      = delete;

  registered_object_guard(registered_object_guard&&) = default;

  /**
   * @brief Get's a constant reference to the stored object.
   * @return ObjectT const&
   */
  [[nodiscard]] ObjectT const& get() const
  {
    GQE_EXPECTS(_registered_object, "Calling get on empty object", std::runtime_error);
    return _registered_object->get();
  }

  /**
   * @brief Get's a reference to the stored object.
   * @return ObjectT&
   */
  [[nodiscard]] ObjectT& get()
  {
    GQE_EXPECTS(_registered_object, "Calling get on empty object", std::runtime_error);
    return _registered_object->get();
  }

  /**
   * @brief Checks if the internal registeredObject is valid.
   * @return true If valid.
   */
  [[nodiscard]] explicit operator bool() const noexcept { return _registered_object != nullptr; }

 private:
  std::shared_ptr<registry_object_type> _registered_object;  //< The registered object.
};

/**
 * @brief An object_registry is a thread-safe map of objects registered
 *  by unique key. External access to each object is guarded from
 *  concurrent access by a registered_object_guard.
 * @tparam KeyT The registry's key type.
 * @tparam ObjectT The registry's object type.
 */
template <typename KeyT, typename ObjectT>
class object_registry {
 public:
  using registry_object_type = detail::registered_object<ObjectT>;
  using guard_type           = registered_object_guard<ObjectT>;

  /**
   * @returns True if the registry contains the key, otherwise false.
   */
  [[nodiscard]] bool contains(KeyT key) const
  {
    std::unique_lock lock(_mutex);
    auto iter = _objects_by_key.find(key);
    return iter != _objects_by_key.end();
  }

  /**
   * @brief Probes the registry for an object registered to \a key.
   * @param[in] key The probe key.
   * @return A registered_object_guard containing the object or an empty registered_object_guard
   *  if no object was found.
   */
  [[nodiscard]] guard_type find(KeyT key)
  {
    std::unique_lock lock(_mutex);

    auto iter = _objects_by_key.find(key);
    if (iter == _objects_by_key.end()) { return guard_type(nullptr); }

    auto registered_object = iter->second;

    return guard_type(registered_object);
  }

  /**
   * @brief Registers \a object at \a key.
   * @param[in] key The registration key of the object.
   * @param[in] object The object being registered.
   */
  void register_object(KeyT key, ObjectT object)
  {
    std::unique_lock lock(_mutex);

    auto results_pair =
      _objects_by_key.insert({key, std::make_shared<registry_object_type>(std::move(object))});

    GQE_EXPECTS(results_pair.second, "An object is already registered at key.", std::runtime_error);
  }

  /**
   * @brief Unregisters \a object at \a key.
   * @param[in] key The registration key of the object.
   */
  void unregister_object(KeyT key)
  {
    std::unique_lock lock(_mutex);

    auto iter = _objects_by_key.find(key);
    GQE_EXPECTS(
      iter != _objects_by_key.end(), "No object is registered at key.", std::runtime_error);

    auto registered_object = iter->second;

    // Wait for exclusive access to the object.
    std::unique_lock obj_lock(*registered_object);
    _objects_by_key.erase(key);
  }

 private:
  std::unordered_map<KeyT, std::shared_ptr<registry_object_type>>
    _objects_by_key;          //< A map of registry_object_types by their key.
  mutable std::mutex _mutex;  //< A mutex to prevent concurrent access to the registry.
};

}  // namespace gqe::executor_next
